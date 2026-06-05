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

"""Per-iteration schedulers for :class:`~susan.project.SubtomoAvgSched`.

Each scheduler group is a class whose nested classes are callable objects
(``__call__``).  Schedulers are optional — leaving a slot ``None`` on
:class:`~susan.project.SubtomoAvgSched` means that attribute is left
unchanged by ``run()``.

Groups
------
Bandpass
    Control ``aligner.bandpass`` and ``ctf_refiner.bandpass`` per iteration.
AngularSearch
    Control the angular search parameters passed to
    ``aligner.set_angular_search`` per iteration.
IterationType
    Control ``iteration_type`` per iteration.
"""

import numpy as _np
import copy   as _copy


###########################################################################
# Bandpass schedulers
###########################################################################

class Bandpass:
    """Namespace for bandpass scheduler variants.

    All variants share the same call signature::

        new_bp = scheduler(ite, bp_est, current_bandpass)

    Parameters
    ----------
    ite : int
        Current iteration number (1-based).
    bp_est : float or array-like
        Resolution estimate in Fourier pixels from the previous iteration's
        FSC.  If an array (MRA case) it is reduced to a scalar according to
        the scheduler's ``reduce`` policy (``Bandpass.Adaptive``; default
        ``'max'``).
    current_bandpass : bandpass object
        The bandpass object to use as template (its ``highpass``, ``lowpass``,
        and ``rolloff`` attributes are read).  A modified **copy** is
        returned; the original is never mutated.

    Returns
    -------
    bandpass object
        A new bandpass object with updated ``lowpass`` (and ``highpass`` /
        ``rolloff`` left unchanged or set per scheduler logic).
    """

    @staticmethod
    def _scalar_bp_est(bp_est, reduce='max'):
        """Collapse a (possibly per-reference) resolution estimate to a scalar.

        *reduce* selects the reduction for the MRA (array) case:
        ``'max'``, ``'min'``, or ``'mean'`` (aliases ``'average'`` / ``'avg'``).
        A scalar *bp_est* is returned unchanged.
        """
        funcs = {'max': _np.max, 'min': _np.min,
                 'mean': _np.mean, 'average': _np.mean, 'avg': _np.mean}
        r = str(reduce).lower()
        if r not in funcs:
            raise ValueError(
                "Invalid reduce '%s' (accepted: 'max', 'min', 'mean', "
                "'average', 'avg')." % reduce)
        return float(funcs[r](_np.asarray(bp_est, dtype=float)))

    @staticmethod
    def _fp_max(box_size, rolloff, user_fp_max=None):
        limit = box_size // 2 - rolloff
        if user_fp_max is not None:
            return min(int(user_fp_max), limit)
        return limit

    # ------------------------------------------------------------------

    class Fixed:
        """Return a copy of *current_bandpass* with ``lowpass`` clamped to
        ``fp_max``.  The ``lowpass`` set on the aligner before the loop is
        used unchanged (only clamping is applied).

        Parameters
        ----------
        box_size : int
        fp_max : int, optional
            Hard ceiling on ``lowpass`` in Fourier pixels.  Defaults to
            ``box_size // 2 - rolloff`` (computed from the bandpass rolloff
            at call time).
        """

        def __init__(self, box_size, fp_max=None):
            self._box_size   = box_size
            self._user_fp_max = fp_max

        def __call__(self, ite, bp_est, bandpass):
            bp   = _copy.copy(bandpass)
            fmax = Bandpass._fp_max(self._box_size, bp.rolloff, self._user_fp_max)
            bp.lowpass = min(bp.lowpass, fmax)
            return bp

        def __repr__(self):
            return 'Bandpass.Fixed(box_size=%d)' % self._box_size

    # ------------------------------------------------------------------

    class Adaptive:
        """Chase the FSC-estimated resolution each iteration.

        ``lowpass = min(bp_est, fp_max)``.  An optional ``max_step`` limits
        how much ``lowpass`` can increase in a single iteration relative to
        the *current* bandpass.

        Parameters
        ----------
        box_size : int
        max_step : float, optional
            Maximum increase in ``lowpass`` per iteration.  ``None`` means
            unlimited.
        fp_max : int, optional
            Hard ceiling.  Defaults to ``box_size // 2 - rolloff``.
        reduce : str, optional
            Reduction applied to a per-reference (MRA) ``bp_est`` array before
            it is used as a single global ``lowpass``: ``'max'`` (default,
            chase the best-resolving class), ``'min'`` (limit to the
            worst-resolving class), or ``'mean'`` (aliases ``'average'`` /
            ``'avg'``).  Ignored for a scalar ``bp_est``.
        """

        def __init__(self, box_size, max_step=None, fp_max=None, reduce='max'):
            self._box_size    = box_size
            self._max_step    = max_step
            self._user_fp_max = fp_max
            self._reduce      = reduce

        def __call__(self, ite, bp_est, bandpass):
            bp    = _copy.copy(bandpass)
            fmax  = Bandpass._fp_max(self._box_size, bp.rolloff, self._user_fp_max)
            new_lp = min(Bandpass._scalar_bp_est(bp_est, self._reduce), fmax)
            if self._max_step is not None:
                new_lp = min(new_lp, bp.lowpass + self._max_step)
            bp.lowpass = new_lp
            return bp

        def __repr__(self):
            return 'Bandpass.Adaptive(box_size=%d, max_step=%s, reduce=%r)' % (
                self._box_size, self._max_step, self._reduce)

    # ------------------------------------------------------------------

    class Ramp:
        """Increase ``lowpass`` by a fixed step each iteration, capped at
        ``fp_max``.  ``bp_est`` is ignored.

        Parameters
        ----------
        box_size : int
        step : float, optional
            Increase per iteration.  Default: ``2.5``.
        fp_max : int, optional
            Hard ceiling.  Defaults to ``box_size // 2 - rolloff``.
        """

        def __init__(self, box_size, step=2.5, fp_max=None):
            self._box_size    = box_size
            self._step        = step
            self._user_fp_max = fp_max

        def __call__(self, ite, bp_est, bandpass):
            bp   = _copy.copy(bandpass)
            fmax = Bandpass._fp_max(self._box_size, bp.rolloff, self._user_fp_max)
            bp.lowpass = min(bp.lowpass + self._step, fmax)
            return bp

        def __repr__(self):
            return 'Bandpass.Ramp(box_size=%d, step=%g)' % (
                self._box_size, self._step)


###########################################################################
# Angular search schedulers
###########################################################################

class AngularSearch:
    """Namespace for angular search scheduler variants.

    All variants share the call signature::

        cone_span, cone_step, inplane_span, inplane_step = scheduler(ite, lp)

    Parameters
    ----------
    ite : int
        Current iteration number (1-based).
    lp : float
        Current ``lowpass`` in Fourier pixels (after bandpass scheduling).

    Returns
    -------
    tuple of float
        ``(cone_span, cone_step, inplane_span, inplane_step)`` — suitable
        for ``aligner.set_angular_search(*result)``.

    .. note::
        Schedulers in this group do **not** touch ``aligner.refine.levels``
        or ``aligner.refine.factor``; set those manually.
    """

    # ------------------------------------------------------------------

    class Fixed:
        """Return constant angular search parameters.

        Parameters
        ----------
        cone_span, cone_step, inplane_span, inplane_step : float
        """

        def __init__(self, cone_span, cone_step, inplane_span, inplane_step):
            self._params = (float(cone_span), float(cone_step),
                            float(inplane_span), float(inplane_step))

        def __call__(self, ite, lp):
            return self._params

        def __repr__(self):
            return 'AngularSearch.Fixed(%g, %g, %g, %g)' % self._params

    # ------------------------------------------------------------------

    class Adaptive:
        """Both cone and inplane spans scale with the current lowpass.

        ``span = factor * arctan2(step_pix, lp)`` (degrees).

        Parameters
        ----------
        cone_factor : float, optional
            Multiplier for the cone span.  Default: ``4``.
        cone_step : float, optional
            Step in degrees for the cone search (also used to compute span
            via ``arctan2``).  Default: ``1``.
        inplane_factor : float, optional
            Multiplier for the inplane span.  Default: ``4``.
        inplane_step : float, optional
            Step in degrees for the inplane search.  Default: ``1``.
        skip_first : bool, optional
            If ``True``, return ``(0, 1, 0, 1)`` when ``ite == 1``.
            Default: ``True``.
        """

        def __init__(self, cone_factor=4, cone_step=1,
                     inplane_factor=4, inplane_step=1, skip_first=True):
            self._cf  = cone_factor
            self._cs  = float(cone_step)
            self._if  = inplane_factor
            self._is  = float(inplane_step)
            self._skip = skip_first

        def __call__(self, ite, lp):
            if self._skip and ite == 1:
                return (0.0, 1.0, 0.0, 1.0)
            ang = float(_np.rad2deg(_np.arctan2(1.0, lp)))
            return (self._cf * ang, self._cs,
                    self._if * ang, self._is)

        def __repr__(self):
            return ('AngularSearch.Adaptive(cone_factor=%g, cone_step=%g, '
                    'inplane_factor=%g, inplane_step=%g, skip_first=%s)'
                    % (self._cf, self._cs, self._if, self._is, self._skip))

    # ------------------------------------------------------------------

    class ConeOnly:
        """Only the cone span adapts; inplane is fixed.

        Parameters
        ----------
        cone_factor : float, optional
            Default: ``4``.
        cone_step : float, optional
            Default: ``1``.
        inplane_span : float, optional
            Fixed inplane span.  Default: ``0``.
        inplane_step : float, optional
            Default: ``1``.
        skip_first : bool, optional
            Default: ``True``.
        """

        def __init__(self, cone_factor=4, cone_step=1,
                     inplane_span=0, inplane_step=1, skip_first=True):
            self._cf   = cone_factor
            self._cs   = float(cone_step)
            self._is_  = float(inplane_span)
            self._ist  = float(inplane_step)
            self._skip = skip_first

        def __call__(self, ite, lp):
            if self._skip and ite == 1:
                return (0.0, 1.0, self._is_, self._ist)
            ang = float(_np.rad2deg(_np.arctan2(1.0, lp)))
            return (self._cf * ang, self._cs, self._is_, self._ist)

        def __repr__(self):
            return ('AngularSearch.ConeOnly(cone_factor=%g, cone_step=%g, '
                    'inplane_span=%g, inplane_step=%g, skip_first=%s)'
                    % (self._cf, self._cs, self._is_, self._ist, self._skip))

    # ------------------------------------------------------------------

    class InplaneOnly:
        """Only the inplane span adapts; cone is fixed.

        Parameters
        ----------
        cone_span : float, optional
            Fixed cone span.  Default: ``0``.
        cone_step : float, optional
            Default: ``1``.
        inplane_factor : float, optional
            Default: ``4``.
        inplane_step : float, optional
            Default: ``1``.
        skip_first : bool, optional
            Default: ``True``.
        """

        def __init__(self, cone_span=0, cone_step=1,
                     inplane_factor=4, inplane_step=1, skip_first=True):
            self._cs_  = float(cone_span)
            self._cst  = float(cone_step)
            self._if   = inplane_factor
            self._is   = float(inplane_step)
            self._skip = skip_first

        def __call__(self, ite, lp):
            if self._skip and ite == 1:
                return (self._cs_, self._cst, 0.0, 1.0)
            ang = float(_np.rad2deg(_np.arctan2(1.0, lp)))
            return (self._cs_, self._cst, self._if * ang, self._is)

        def __repr__(self):
            return ('AngularSearch.InplaneOnly(cone_span=%g, cone_step=%g, '
                    'inplane_factor=%g, inplane_step=%g, skip_first=%s)'
                    % (self._cs_, self._cst, self._if, self._is, self._skip))

    # ------------------------------------------------------------------

    class AlternatingConeInplane:
        """Alternate between cone-only and inplane-only search each call.

        On "cone" phases the inplane span is zero; on "inplane" phases the
        cone span is zero.

        Parameters
        ----------
        cone_factor : float, optional
            Default: ``4``.
        cone_step : float, optional
            Default: ``1``.
        inplane_factor : float, optional
            Default: ``4``.
        inplane_step : float, optional
            Default: ``1``.
        start_with : {'cone', 'inplane'}, optional
            Which search to perform first.  Default: ``'cone'``.
        sync_to_ite : bool, optional
            If ``True``, phase = ``(ite - 1) % 2`` (cone on odd iterations
            when ``start_with='cone'``).  If ``False``, an internal counter
            is incremented each call and can be reset with :meth:`reset`.
            Default: ``True``.
        """

        def __init__(self, cone_factor=4, cone_step=1,
                     inplane_factor=4, inplane_step=1,
                     start_with='cone', sync_to_ite=True):
            self._cf        = cone_factor
            self._cs        = float(cone_step)
            self._if        = inplane_factor
            self._is        = float(inplane_step)
            self._start_cone = (start_with.lower() == 'cone')
            self._sync      = sync_to_ite
            self._counter   = 0

        def reset(self, phase=0):
            """Reset the internal counter.

            Parameters
            ----------
            phase : int, optional
                Starting phase (0 = ``start_with``, 1 = the other).
                Default: ``0``.
            """
            self._counter = int(phase)

        def _is_cone_phase(self, ite):
            if self._sync:
                phase = (ite - 1) % 2
            else:
                phase = self._counter % 2
                self._counter += 1
            # phase 0 → start_with, phase 1 → the other
            return self._start_cone if phase == 0 else not self._start_cone

        def __call__(self, ite, lp):
            ang = float(_np.rad2deg(_np.arctan2(1.0, lp)))
            if self._is_cone_phase(ite):
                return (self._cf * ang, self._cs, 0.0, 1.0)
            else:
                return (0.0, 1.0, self._if * ang, self._is)

        def __repr__(self):
            start = 'cone' if self._start_cone else 'inplane'
            return ('AngularSearch.AlternatingConeInplane('
                    'cone_factor=%g, cone_step=%g, '
                    'inplane_factor=%g, inplane_step=%g, '
                    'start_with=%r, sync_to_ite=%s)'
                    % (self._cf, self._cs, self._if, self._is,
                       start, self._sync))


###########################################################################
# Iteration type schedulers
###########################################################################

class IterationType:
    """Namespace for iteration-type scheduler variants.

    All variants share the call signature::

        type_str = scheduler(ite)

    Parameters
    ----------
    ite : int
        Current iteration number (1-based).

    Returns
    -------
    str
        One of ``'3D'``, ``'2D'``, or ``'ctf'``.
    """

    # ------------------------------------------------------------------

    class Fixed:
        """Always return the same iteration type.

        Parameters
        ----------
        type_ : str
            ``'3D'``, ``'2D'``, or ``'ctf'``.
        """

        def __init__(self, type_):
            self._type = str(type_)

        def __call__(self, ite):
            return self._type

        def __repr__(self):
            return 'IterationType.Fixed(%r)' % self._type

    # ------------------------------------------------------------------

    class Alternating:
        """Alternate between two types based on iteration parity or an
        internal counter.

        Parameters
        ----------
        odd : str, optional
            Type for odd iterations (or odd counter values).  Default:
            ``'ctf'``.
        even : str, optional
            Type for even iterations (or even counter values).  Default:
            ``'2D'``.
        sync_to_ite : bool, optional
            If ``True``, parity is determined by ``ite % 2``.  If ``False``,
            an internal counter is used.  Default: ``True``.
        """

        def __init__(self, odd='ctf', even='2D', sync_to_ite=True):
            self._odd   = str(odd)
            self._even  = str(even)
            self._sync  = sync_to_ite
            self._counter = 0

        def reset(self, phase=0):
            """Reset the internal counter.

            Parameters
            ----------
            phase : int, optional
                0 → next call returns the "even" value; 1 → "odd".
                Default: ``0``.
            """
            self._counter = int(phase)

        def __call__(self, ite):
            if self._sync:
                return self._odd if (ite % 2) == 1 else self._even
            val = self._odd if (self._counter % 2) == 1 else self._even
            self._counter += 1
            return val

        def __repr__(self):
            return ('IterationType.Alternating(odd=%r, even=%r, sync_to_ite=%s)'
                    % (self._odd, self._even, self._sync))

    # ------------------------------------------------------------------

    class Cyclic:
        """Cycle through a sequence of iteration types.

        Parameters
        ----------
        sequence : list of str
            Ordered list of types to cycle through, e.g.
            ``['3D', '3D', 'ctf', '2D']``.
        sync_to_ite : bool, optional
            If ``True``, index = ``(ite - 1) % len(sequence)``.  If
            ``False``, an internal counter is used.  Default: ``True``.
        """

        def __init__(self, sequence, sync_to_ite=True):
            if not sequence:
                raise ValueError('sequence must not be empty')
            self._seq     = list(sequence)
            self._sync    = sync_to_ite
            self._counter = 0

        def reset(self, phase=0):
            """Reset the internal counter.

            Parameters
            ----------
            phase : int, optional
                Starting index into the sequence.  Default: ``0``.
            """
            self._counter = int(phase) % len(self._seq)

        def __call__(self, ite):
            if self._sync:
                return self._seq[(ite - 1) % len(self._seq)]
            val = self._seq[self._counter % len(self._seq)]
            self._counter += 1
            return val

        def __repr__(self):
            return ('IterationType.Cyclic(%r, sync_to_ite=%s)'
                    % (self._seq, self._sync))
