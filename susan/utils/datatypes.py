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

from dataclasses import dataclass as _dc

@_dc
class fsc_info:
    """Resolution estimate returned by :func:`susan.utils.fsc_analyse`.

    .. rubric:: Attributes

    Attributes
    ----------
    fpix : float
        Resolution in Fourier pixels.
    res : float
        Resolution in Ångstroms.  ``0.0`` if the FSC never drops below the
        requested threshold.
    """
    fpix: float
    res:  float

@_dc
class ssnr:
    """Ad-hoc SSNR model used by :class:`susan.modules.Aligner` and
    :class:`susan.modules.Averager`.

    The spectral signal-to-noise ratio is modelled as:

    .. math::
        SSNR(s) = 10^{3S} \\cdot e^{-100 F s}

    where *s* is the spatial frequency in 1/Angstrom (:math:`s = r/(N \\cdot
    apix)`, with *r* the Fourier-pixel radius and *N* the box size).  This is
    the SSNR of a single projection, not of the reconstructed map: the term
    :math:`1/SSNR(s)` is added to the Wiener denominator once per projection,
    so the resulting filter does not depend on how many particles or tilts
    contribute.  See the
    `Ad-hoc SSNR <https://www.nature.com/articles/s41592-019-0580-y#Sec19>`_
    definition for background.

    .. rubric:: Attributes

    Attributes
    ----------
    S : float
        Strength parameter: the SSNR at zero frequency is :math:`10^{3S}`.
        Typically set to 1.
    F : float
        Fall-off parameter, typically in the range 0 – 0.5.  The SSNR loses
        one decade every :math:`0.023/F` 1/Angstrom.  Its value depends on
        the noise level of the data.  With ``S = 1``, the Wiener filter
        reaches half power at roughly :math:`16 \\cdot F` Angstrom.
    """
    # SSNR(s) = (10^(3*S)) * exp( -s * 100*F ),  s in 1/Angstrom
    S: float
    F: float

@_dc
class bandpass:
    """Bandpass filter descriptor with cosine-decay edges.

    The filter is flat between ``highpass`` and ``lowpass`` and uses a
    cosine rolloff of width ``rolloff`` at both edges.

    .. image:: ../images/bandpass_def_light.svg
        :align: center
        :class: only-light

    .. image:: ../images/bandpass_def_dark.svg
        :align: center
        :class: only-dark

    .. rubric:: Attributes

    Attributes
    ----------
    highpass : float
        High-pass cutoff in Fourier pixels.
    lowpass : float
        Low-pass cutoff in Fourier pixels.
    rolloff : float
        Width of the cosine taper at each edge, in Fourier pixels.
    """
    highpass: float
    lowpass:  float
    rolloff:  float

@_dc
class search_params:
    """Generic angular or offset search range and step.

    .. rubric:: Attributes

    Attributes
    ----------
    span : float
        Total search range (half-range is ``span / 2``).
    step : float
        Step size of the search.
    """
    span: float
    step: float

@_dc
class offset_params:
    """Offset (translation) search parameters.

    .. image:: ../images/offsets_def_light.svg
        :align: center
        :class: only-light

    .. image:: ../images/offsets_def_dark.svg
        :align: center
        :class: only-dark

    .. rubric:: Attributes

    Attributes
    ----------
    span : list of float
        Offset range in 3-D, ordered ``[X, Y, Z]``.
    step : float
        Offset step size in pixels.  For 3-D searches defaults to 1 but
        also accepts sub-pixel values such as 0.5 or 0.25.  For 2-D
        searches it is fixed to 1.
    kind : str
        Shape of the search volume.  Valid options for 3-D searches:
        ``'ellipsoid'`` (default), ``'cylinder'``, ``'cuboid'``.
        For 2-D searches only ``'circle'`` is accepted.
    """
    span: list
    step: float
    kind: str

@_dc
class refine_params:
    """Multi-level angular refinement parameters.

    Modelled after `DYNAMO's angular refinement policy
    <https://www.dynamo-em.org/w/index.php?title=Multilevel_refinement>`_.

    .. rubric:: Attributes

    Attributes
    ----------
    levels : int
        Number of refinement levels.  Each level halves the angular step.
    factor : int
        Range multiplier per level.  Given the angular step *ang* at the
        current level, the search range is ±``factor`` · *ang*.
    """
    levels: int
    factor: int

@_dc
class range_params:
    """Integer range descriptor.

    .. rubric:: Attributes

    Attributes
    ----------
    min_val : int
        Minimum value (inclusive).
    max_val : int
        Maximum value (inclusive).
    """
    min_val: int
    max_val: int

class mpi_params:
    """Configuration for launching MPI-parallel processing modules.

    The MPI binary is invoked roughly as::

        cmd = mpi_params.cmd % mpi_params.arg + ' susan_aligner_mpi ' + aligner.get_args()
        os.system(cmd)

    .. rubric:: Attributes

    Attributes
    ----------
    cmd : str or None
        Base MPI launch command, e.g. ``'srun -n %d'`` (SLURM) or
        ``'mpirun -n %d'`` (generic MPI).  ``None`` disables MPI.
    arg : int or None
        Argument substituted into ``cmd`` (typically the number of nodes).
        Defaults to ``None`` (no substitution).
    """

    cmd: str
    arg: int

    def __init__(self, cmd=None, arg=None):
        self.cmd = cmd
        self.arg = arg

    def gen_cmd(self):
        """Build the complete MPI prefix string.

        Returns
        -------
        str
            Empty string if ``cmd`` is ``None`` or empty.  ``cmd`` alone if
            ``arg`` is ``None``.  Otherwise ``cmd % arg`` followed by a
            trailing space.
        """
        if self.cmd is None or not self.cmd:
            return ""
        if self.arg is None or not self.arg:
                return self.cmd
        return self.cmd%self.arg + ' '

@_dc
class inversion_params:
    """Parameters for iterative inversion of the sampling function.

    .. note::
        This is an advanced parameter.  The defaults work well in most cases;
        change only if you understand the sampling-function correction.

    Used to correct for non-uniform angular sampling in reconstruction.
    See `Pipe & Menon (1999)
    <https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V>`_.

    .. rubric:: Attributes

    Attributes
    ----------
    ite : int
        Number of iterations.  Default: 10.
    std : float
        Standard deviation of the Gaussian used to approximate the gridding
        kernel.  Values of ``0`` or less select it from the gridding method:
        ``0.41`` for linear, ``0.69`` for Kaiser-Bessel, ``0.74`` when
        splatting.  Default: ``-1`` (automatic).
    """
    ite: int
    std: float

@_dc
class boost_lowfreq_params:
    """Low-frequency boost for reconstructions with few particles.

    .. warning::
        This is an experimental feature.  Results may change in future versions
        and the parameter defaults have not been validated on all data types.

    Applies an extra multiplicative weight to the lowest spatial frequencies
    before reconstruction.

    .. rubric:: Attributes

    Attributes
    ----------
    scale : float
        Strength of the boost.  Typically 5–10.
    value : float
        Frequency up to which the boost is applied, in Fourier pixels.
        Typically 2–5.
    decay : float
        Width of the cosine decay that tapers the boost to zero.
        Typically 3–6.
    """
    scale: float
    value: float
    decay: float
