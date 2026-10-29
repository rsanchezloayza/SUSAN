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
``output_type='alignment_stats'`` (:class:`susan.modules.Aligner`).

Keys (numpy arrays):

``projections/stats`` (n_ptcl, K, NF)
    Per particle (rows of the aligner's input particles) and projection, the fields :data:`FIELDS` of the raw CC
    (no projection weight, no priors) over the offsets at the input orientation; NaN where undefined.
``projections/fields`` (NF,)
    The field names.
``grouped<gs>/stats`` (n_tomo, G, K, NF), ``grouped<gs>/count`` (n_tomo, G, K)
    The same fields for the mean CC map of each group at the input orientation, and the number of particles in it; groups: gs x gs equal-count bins of the particle x, y per tomogram, G = halves * gs * gs.
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
"""

__all__ = ['FIELDS', 'path', 'read']

import numpy as _np

FIELDS = ('cc', 'px', 'py', 'z_peak', 'h_xx', 'h_xy', 'h_yy', 'dx2', 'dy2', 'r2')


def path(ptcls_out):
    """Statistics file of an aligner output: the ``.ptclsraw`` extension replaced by ``.stats.npz``."""
    base = ptcls_out[:-len('.ptclsraw')] if ptcls_out.endswith('.ptclsraw') else ptcls_out
    return base + '.stats.npz'


def read(filename):
    """Load a ``.stats.npz`` (or the statistics of a ``.ptclsraw`` output) into a dict of arrays (keys above)."""
    if filename.endswith('.ptclsraw'):
        filename = path(filename)
    with _np.load(filename) as f:
        return {k: f[k] for k in f.files}
