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

"""Scheduler-driven subtomogram averaging loop.

.. warning::

   **Experimental / in development.**  The scheduler protocol, the set of
   built-in schedulers in :mod:`susan.project.Schedulers`, and the factory
   presets (``make_*_refinement``) are still being shaped and may change
   between releases.  Use :class:`~susan.project.SubtomoAvg.SubtomoAvg`
   directly for stable manual loops.

:class:`SubtomoAvgSched` extends :class:`~susan.project.SubtomoAvg.SubtomoAvg`
with three optional scheduler slots and a :meth:`~SubtomoAvgSched.run` method
that drives the iteration loop.  All schedulers are optional — a ``None``
slot means that attribute is left at whatever the user set manually.

Factory classmethods (:meth:`~SubtomoAvgSched.make_3d_refinement`,
:meth:`~SubtomoAvgSched.make_2d_refinement`,
:meth:`~SubtomoAvgSched.make_mixed_ctf_2d`) return pre-configured instances.
Because project state lives on disk, any preset can continue from the results
of any other preset for the same project directory.
"""

import numpy as _np

from susan.project.SubtomoAvg import SubtomoAvg as _SubtomoAvg
from susan.project import Schedulers as _S


class SubtomoAvgSched(_SubtomoAvg):
    """Subtomogram averaging with optional per-iteration schedulers.

    .. warning::

       **Experimental / in development.**  The scheduler protocol and the
       factory presets may change between releases.  Use
       :class:`~susan.project.SubtomoAvg.SubtomoAvg` directly for stable
       manual loops.

    Extends :class:`~susan.project.SubtomoAvg.SubtomoAvg` with three
    scheduler slots and a :meth:`run` loop.  Leaving a scheduler as
    ``None`` keeps that setting unchanged each iteration.

    Parameters
    ----------
    prj_name : str
        Project directory.
    box_size : int, optional
        Box size in pixels.  Reads ``info.prjtxt`` when omitted.

    Attributes
    ----------
    bandpass_scheduler : callable or None
        Signature: ``new_bp = sched(ite, bp_est, current_bandpass)``.
        Result is assigned to **both** ``aligner.bandpass`` and
        ``ctf_refiner.bandpass``.  Default: ``None``.
    angular_search_scheduler : callable or None
        Signature: ``(cs, css, is_, iss) = sched(ite, lp)``.
        Result is unpacked into ``aligner.set_angular_search(*result)``.
        Default: ``None``.
    iteration_type_scheduler : callable or None
        Signature: ``type_str = sched(ite)``.
        Result is assigned to ``self.iteration_type``.  Default: ``None``.
    """

    def __init__(self, prj_name, box_size=None):
        super().__init__(prj_name, box_size)
        self.bandpass_scheduler       = None
        self.angular_search_scheduler = None
        self.iteration_type_scheduler = None

    # ------------------------------------------------------------------
    # Scheduler application helpers
    # ------------------------------------------------------------------

    def _apply_schedulers(self, ite, bp_est):
        if self.iteration_type_scheduler is not None:
            self.iteration_type = self.iteration_type_scheduler(ite)

        lp = self.aligner.bandpass.lowpass
        if self.bandpass_scheduler is not None:
            new_bp = self.bandpass_scheduler(ite, bp_est, self.aligner.bandpass)
            self.aligner.bandpass     = new_bp
            self.ctf_refiner.bandpass = new_bp
            lp = new_bp.lowpass

        if self.angular_search_scheduler is not None:
            self.aligner.set_angular_search(
                *self.angular_search_scheduler(ite, lp))

    def _reset_schedulers(self):
        for sched in (self.bandpass_scheduler,
                      self.angular_search_scheduler,
                      self.iteration_type_scheduler):
            if sched is not None and hasattr(sched, 'reset'):
                sched.reset()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, start, end, reset=False):
        """Run iterations from *start* to *end* (inclusive).

        Schedulers are applied **before** each iteration.  ``bp_est``
        passed to the bandpass scheduler is the resolution estimate
        (Fourier pixels) returned by the *previous* iteration (or
        ``aligner.bandpass.lowpass`` for the very first call).

        Parameters
        ----------
        start : int
            First iteration number.
        end : int
            Last iteration number (inclusive).
        reset : bool, optional
            If ``True``, call ``reset()`` on all schedulers that support it
            before starting the loop.  Default: ``False``.

        Returns
        -------
        list of float or numpy.ndarray
            Per-iteration resolution estimates in Fourier pixels, one entry
            per iteration.
        """
        if reset:
            self._reset_schedulers()

        if self._validate_iteration_type() == 'ctf':
            bp_est = self.ctf_refiner.bandpass.lowpass
        else:
            bp_est = self.aligner.bandpass.lowpass
        results = []

        for ite in range(start, end + 1):
            self._apply_schedulers(ite, bp_est)
            bp_est = self.run_iteration(ite)
            results.append(bp_est)

        return results

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def make_3d_refinement(cls, prj_name, box_size=None, *,
                           lp_init=30.0, lp_max=None, lp_step=2.5,
                           cone_factor=4, inplane_factor=4,
                           tomogram_file=None, initial_reference=None,
                           initial_particles=None):
        """3-D closed-loop refinement with adaptive angular search.

        Bandpass chases the FSC estimate (clamped by *lp_max*); angular
        search spans scale with the current lowpass.  First iteration uses
        no angular search (``skip_first=True``).

        Parameters
        ----------
        prj_name : str
        box_size : int, optional
        lp_init : float, optional
            Starting lowpass in Fourier pixels.  Default: ``30``.
        lp_max : float, optional
            Bandpass ceiling.  Defaults to ``box_size // 2 - rolloff``.
        lp_step : float, optional
            Maximum lowpass increase per iteration.  Default: ``2.5``.
        cone_factor, inplane_factor : float, optional
            Angular span multipliers.  Default: ``4``.
        tomogram_file, initial_reference, initial_particles : str, optional
            Written to ``info.prjtxt`` only when provided (existing project
            files are left untouched otherwise).

        Returns
        -------
        SubtomoAvgSched
        """
        sta = cls(prj_name, box_size)
        if tomogram_file:     sta.tomogram_file     = tomogram_file
        if initial_reference: sta.initial_reference = initial_reference
        if initial_particles: sta.initial_particles = initial_particles

        sta.iteration_type_scheduler  = _S.IterationType.Fixed('3D')
        sta.bandpass_scheduler        = _S.Bandpass.Adaptive(
            sta.box_size, max_step=lp_step, fp_max=lp_max)
        sta.angular_search_scheduler  = _S.AngularSearch.Adaptive(
            cone_factor=cone_factor, inplane_factor=inplane_factor)

        sta.aligner.bandpass.lowpass  = lp_init
        return sta

    @classmethod
    def make_2d_refinement(cls, prj_name, box_size=None, *,
                           lp_init=45.0, lp_max=None, lp_step=2.5,
                           inplane_factor=4,
                           tomogram_file=None, initial_reference=None,
                           initial_particles=None):
        """2-D in-plane refinement with adaptive inplane angular search.

        Parameters
        ----------
        prj_name : str
        box_size : int, optional
        lp_init : float, optional
            Default: ``45``.
        lp_max : float, optional
        lp_step : float, optional
            Default: ``2.5``.
        inplane_factor : float, optional
            Default: ``4``.
        tomogram_file, initial_reference, initial_particles : str, optional

        Returns
        -------
        SubtomoAvgSched
        """
        sta = cls(prj_name, box_size)
        if tomogram_file:     sta.tomogram_file     = tomogram_file
        if initial_reference: sta.initial_reference = initial_reference
        if initial_particles: sta.initial_particles = initial_particles

        sta.iteration_type_scheduler  = _S.IterationType.Fixed('2D')
        sta.bandpass_scheduler        = _S.Bandpass.Adaptive(
            sta.box_size, max_step=lp_step, fp_max=lp_max)
        sta.angular_search_scheduler  = _S.AngularSearch.InplaneOnly(
            inplane_factor=inplane_factor)

        sta.aligner.bandpass.lowpass  = lp_init
        return sta

    @classmethod
    def make_mixed_ctf_2d(cls, prj_name, box_size=None, *,
                          lp_init=45.0, lp_max=None, lp_step=2.5,
                          cone_factor=4, inplane_factor=4,
                          odd='ctf', even='2D',
                          tomogram_file=None, initial_reference=None,
                          initial_particles=None):
        """Interleaved CTF + 2-D refinement with shared bandpass tracking.

        Parameters
        ----------
        prj_name : str
        box_size : int, optional
        lp_init : float, optional
            Default: ``45``.
        lp_max : float, optional
        lp_step : float, optional
            Default: ``2.5``.
        cone_factor, inplane_factor : float, optional
            Default: ``4``.
        odd, even : str, optional
            Iteration types for odd and even iterations.
            Default: ``'ctf'`` / ``'2D'``.
        tomogram_file, initial_reference, initial_particles : str, optional

        Returns
        -------
        SubtomoAvgSched
        """
        sta = cls(prj_name, box_size)
        if tomogram_file:     sta.tomogram_file     = tomogram_file
        if initial_reference: sta.initial_reference = initial_reference
        if initial_particles: sta.initial_particles = initial_particles

        sta.iteration_type_scheduler  = _S.IterationType.Alternating(
            odd=odd, even=even)
        sta.bandpass_scheduler        = _S.Bandpass.Adaptive(
            sta.box_size, max_step=lp_step, fp_max=lp_max)
        sta.angular_search_scheduler  = _S.AngularSearch.Adaptive(
            cone_factor=cone_factor, inplane_factor=inplane_factor)

        sta.aligner.bandpass.lowpass  = lp_init
        return sta
