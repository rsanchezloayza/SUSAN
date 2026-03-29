###########################################################################
# This file is part of the Substack Analysis (SUSAN) framework.
# Copyright (c) 2018-2023 Ricardo Miguel Sanchez Loayza.
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

__all__ = ['read']

import numpy as _np

def read(filename):
    """Read an EM file and return its data as a float32 array.

    Only float32 (EM data type 5) files with machine code 6 (PC/Linux
    little-endian) are supported.  Raises ValueError for any other format.

    Parameters
    ----------
    filename : str
        Path to the EM file.

    Returns
    -------
    data : numpy.ndarray
        Volume array with shape (Z, Y, X), dtype float32.
    """
    em_hdr = _np.fromfile(filename,dtype=_np.uint8 ,count=4)
    em_siz = _np.fromfile(filename,dtype=_np.uint32,count=3,offset=4)
    if em_hdr[0] != 6 or em_hdr[3] != 5:
       raise ValueError('Invalid EM format.')

    data = _np.fromfile(filename,dtype=_np.float32,count=-1,offset=512)
    data = _np.reshape(data,(em_siz[2],em_siz[1],em_siz[0]))

    return data

