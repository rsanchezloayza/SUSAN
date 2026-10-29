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

"""CTF statistics: the ``<output stem>.ctfstats.npz`` written by the CTF refiner with
``output_type='refinement_stats'`` (:class:`susan.modules.CtfRefiner`).

Per particle and projection, over the linearized coordinate ``v = s^2 D(phi) / D_ref`` (``s`` the spatial frequency,
``D(phi)`` the particle's metadata defocus along the Fourier direction, ``D_ref`` the median metadata defocus of the
projection over the tomogram's particles; ``nb`` bins of width ``vmax / nb``), with ``G = F P exp(2 pi i k.t / box)``
(``F`` the substack, ``P`` the reference projection without CTF, ``t`` the sub-pixel shift of the particle's current
translation, ``k`` in pixels of the padded box); no data whitening.  Keys (numpy arrays), with K the projections of
the tomograms file:

``group<gs>/x``, ``w``, ``s``, ``c`` (n_tomo, G, 2, K, nb)
    Sums of ``Re(G)``, ``|P|^2``, ``|F|^2`` and the pixel count over the particles of each group.  Groups: gs x gs
    equal-count bins of the particle x, y per tomogram, G = halves * gs * gs.  With ``halves`` 2 (independent
    half-sets) the bins are drawn per half and the groups of half h are ``(h-1)*gs*gs .. h*gs*gs - 1``: no group mixes
    the halves.  The ``2`` axis splits the particles by the parity of their index.
``group<gs>/label`` (n_ptcl,)
    Group of each particle (-1: none), an index into G.
``shift/ix``, ``iy``, ``wxx``, ``wyy``, ``wxy`` (n_tomo, halves, 2, K, nb)
    Sums of ``Im(G) kx``, ``Im(G) ky``, ``kx^2 |P|^2``, ``ky^2 |P|^2``, ``kx ky |P|^2`` of all the particles of each
    half-set and index parity, for the per-projection translation bias.
``curves`` (n_ptcl, K, ND)
    Per particle and projection, ``max(sum_v c X, 0) / sqrt(sum_v c^2 W)`` of its own profile, ``c`` the CTF at
    ``D_ref + delta``; NaN where undefined.
``delta`` (ND,), ``dref`` (n_tomo, K)
    The defocus grid (Angstroms, around the metadata defocus) and the reference defocus (0: no particles).
``vmax`` (), ``nb`` (), ``box`` ()
    Range and bins of ``v``, and the padded box size (pixels).
``apix``, ``kv``, ``cs``, ``ac``, ``tomo_id`` (n_tomo,)
    Pixel size, voltage, spherical aberration, amplitude contrast and id of each tomogram.
``halves`` ()
    1, or 2 when the groups are per half-set.
"""

__all__ = ['path', 'read']

import numpy as _np


def path(ptcls_out):
    """Statistics file of a CTF refiner output: the ``.ptclsraw`` extension replaced by ``.ctfstats.npz``."""
    base = ptcls_out[:-len('.ptclsraw')] if ptcls_out.endswith('.ptclsraw') else ptcls_out
    return base + '.ctfstats.npz'


def read(filename):
    """Load a ``.ctfstats.npz`` (or the statistics of a ``.ptclsraw`` output) into a dict of arrays (keys above)."""
    if filename.endswith('.ptclsraw'):
        filename = path(filename)
    with _np.load(filename) as f:
        return {k: f[k] for k in f.files}
