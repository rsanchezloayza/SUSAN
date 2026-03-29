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

__all__ = ['read','write','get_info']

import numpy as _np

def _mode_to_type(mrc_mode):
    if mrc_mode == 0:
        in_type = _np.int8
    elif mrc_mode == 1:
        in_type = _np.int16
    elif mrc_mode == 2:
        in_type = _np.float32
    elif mrc_mode == 6:
        in_type = _np.uint16
    elif mrc_mode == 12:
        in_type = _np.float16
    else:
        raise ValueError
    return in_type

def _type_to_mode(mrc_type):
    if _np.issubdtype(mrc_type,_np.int8):
        out_mode = 0
    elif _np.issubdtype(mrc_type,_np.int16):
        out_mode = 1
    elif _np.issubdtype(mrc_type,_np.float32):
        out_mode = 2
    elif _np.issubdtype(mrc_type,_np.float64):
        out_mode = 2
    elif _np.issubdtype(mrc_type,_np.uint16):
        out_mode = 6
    elif _np.issubdtype(mrc_type,_np.float16):
        out_mode = 12
    else:
        raise ValueError
    return out_mode

def read(filename):
    """Read an MRC/CCP4 file and return its data and pixel size.

    Parameters
    ----------
    filename : str
        Path to the MRC file.

    Returns
    -------
    data : numpy.ndarray
        Volume or image array with shape (Z, Y, X), dtype determined by
        the MRC mode field (float32 for mode 2, int16 for mode 1, etc.).
    pix_size : numpy.ndarray
        Pixel size in Ångströms, shape (3,) float32, ordered (X, Y, Z).
    """
    mrc_shape = _np.fromfile(filename,dtype=_np.uint32 ,count=3)
    mrc_mode  = _np.fromfile(filename,dtype=_np.uint32 ,count=1,offset=12)
    mrc_sampl = _np.fromfile(filename,dtype=_np.uint32 ,count=3,offset=28)
    mrc_cellA = _np.fromfile(filename,dtype=_np.float32,count=3,offset=40)
    mrc_offst = _np.fromfile(filename,dtype=_np.uint32 ,count=1,offset=92)

    pix_size  = (mrc_cellA/mrc_sampl).astype(_np.float32)
    in_type   = _mode_to_type(mrc_mode)

    data = _np.fromfile(filename,dtype=in_type,count=-1,offset=(1024+mrc_offst[0]))
    data = _np.reshape(data,(mrc_shape[2],mrc_shape[1],mrc_shape[0]))

    return data,pix_size

def get_info(filename):
    """Read only the header of an MRC file without loading the data.

    Parameters
    ----------
    filename : str
        Path to the MRC file.

    Returns
    -------
    shape : numpy.ndarray
        Volume dimensions as (NX, NY, NZ), shape (3,) uint32.
    pix_size : numpy.ndarray
        Pixel size in Ångströms, shape (3,) float32, ordered (X, Y, Z).
    dtype : numpy.dtype
        NumPy dtype corresponding to the MRC mode field.
    """
    mrc_shape = _np.fromfile(filename,dtype=_np.uint32 ,count=3)
    mrc_mode  = _np.fromfile(filename,dtype=_np.uint32 ,count=1,offset=12)
    mrc_sampl = _np.fromfile(filename,dtype=_np.uint32 ,count=3,offset=28)
    mrc_cellA = _np.fromfile(filename,dtype=_np.float32,count=3,offset=40)
    
    pix_size  = (mrc_cellA/mrc_sampl).astype(_np.float32)
    in_type   = _mode_to_type(mrc_mode)

    return mrc_shape,pix_size,in_type

def write(data, filename, apix=1, ispg=None, fill_statistics=True):
    """Write a numpy array to an MRC2014 file.

    Parameters
    ----------
    data : numpy.ndarray
        Array to write, shape (Z, Y, X).  float64 is automatically
        downcast to float32.  Supported dtypes: int8, int16, uint16,
        float16, float32, float64.
    filename : str
        Output file path.
    apix : float or array-like of shape (3,), optional
        Pixel size in Ångströms.  A scalar sets all three axes equally.
        Default 1.
    ispg : int, optional
        Space-group number written to the header.  Default 1 (P1).
    fill_statistics : bool, optional
        If True (default), compute and write min, max, mean, and std
        into the header statistics fields.
    """
    apix = _np.array(apix,dtype=_np.float32)
    if apix.size == 1:
        apix = _np.array((apix,apix,apix))
    apix = apix*_np.array(data.shape,dtype=_np.float32)
    hdr  = _np.zeros(256,dtype=_np.uint32)
    apix_uint32 = apix.view(_np.uint32)
    
    if ispg is None:
        ispg = 1
    
    hdr[0]  = data.shape[2]
    hdr[1]  = data.shape[1]
    hdr[2]  = data.shape[0]
    hdr[3]  = _type_to_mode( data.dtype )
    hdr[7]  = data.shape[2]
    hdr[8]  = data.shape[1]
    hdr[9]  = data.shape[0]
    hdr[10]  = apix_uint32[2]
    hdr[11]  = apix_uint32[1]
    hdr[12]  = apix_uint32[0]
    hdr[13] = 1119092736 # 0x42b40000; // 90.0 in hexadecimal notation.
    hdr[14] = 1119092736 # 0x42b40000; // 90.0 in hexadecimal notation.
    hdr[15] = 1119092736 # 0x42b40000; // 90.0 in hexadecimal notation.
    hdr[16] = 1
    hdr[17] = 2
    hdr[18] = 3
    hdr[22] = ispg
    hdr[27] =      20140 # MRC2014 format
    hdr[52] =  542130509 # 0x2050414B ('MAP ')
    hdr[53] =      17476 # 0x00004444 little-endian
    
    if fill_statistics:
        vmin = data.min()
        vmax = data.max()
        vavg = data.mean()
        vstd = data.std()        
        hdr[19] = _np.float32(vmin).view(_np.uint32)
        hdr[20] = _np.float32(vmax).view(_np.uint32)
        hdr[21] = _np.float32(vavg).view(_np.uint32)
        hdr[54] = _np.float32(vstd).view(_np.uint32)
    
    f = open(filename,'wb')
    hdr.tofile(f)
    if data.dtype == 'float64':
        tmp = _np.float32(data)
        tmp.tofile(f)
    else:
        data.tofile(f)
    f.close()

