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

__all__ = ['read','write','get_info','set_statistics','create','write_slices','read_slices']

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

def _header(shape_zyx, dtype, apix=1, ispg=None, origin=None, start=None):
    """MRC2014 header (256 uint32 words) for a (Z, Y, X) volume."""
    apix = _np.array(apix,dtype=_np.float32)
    if apix.size == 1:
        apix = _np.array((apix,apix,apix))
    cell = (apix*_np.array(shape_zyx[::-1],dtype=_np.float32)).astype(_np.float32)
    hdr  = _np.zeros(256,dtype=_np.uint32)
    hdr[0:3]   = shape_zyx[::-1]
    hdr[3]     = _type_to_mode(dtype)
    hdr[7:10]  = shape_zyx[::-1]
    hdr[10:13] = cell.view(_np.uint32)
    hdr[13:16] = 1119092736 # 0x42b40000; // 90.0 in hexadecimal notation.
    hdr[16:19] = (1,2,3)
    hdr[22]    = 1 if ispg is None else ispg
    hdr[27]    =      20140 # MRC2014 format
    hdr[52]    =  542130509 # 0x2050414B ('MAP ')
    hdr[53]    =      17476 # 0x00004444 little-endian
    if start is not None:
        hdr[4:7] = _np.array(start,dtype=_np.int32).view(_np.uint32)
    if origin is not None:
        hdr[49:52] = _np.array(origin,dtype=_np.float32).view(_np.uint32)
    return hdr

def _statistics(v):
    """(min, max, mean, std) float32 of the finite values of v (zeros if none), accumulated in float64."""
    if _np.issubdtype(v.dtype,_np.floating):
        ok = _np.isfinite(v)
        if not ok.all():
            v = v[ok]
    if v.size == 0:
        return _np.zeros(4,dtype=_np.float32)
    return _np.array((v.min(),v.max(),v.mean(dtype=_np.float64),v.std(dtype=_np.float64)),dtype=_np.float32)

def write(data, filename, apix=1, ispg=None, fill_statistics=True, origin=None, start=None):
    """Write a numpy array to an MRC2014 file.

    Parameters
    ----------
    data : numpy.ndarray
        Array to write, shape (Z, Y, X).  float64 is automatically
        downcast to float32.  Supported dtypes: int8, int16, uint16,
        float16, float32, float64.
    filename : str
        Output file path.
    apix : float or array-like of shape (3,), optional, ordered (X, Y, Z).
        Pixel size in Ångströms.  A scalar sets all three axes equally.
        Default 1.
    ispg : int, optional
        Space-group number written to the header.  Default 1 (P1).
    fill_statistics : bool, optional
        If True (default), compute and write min, max, mean, and std
        into the header statistics fields (finite values only, so NaN
        voxels are skipped).
    origin : array-like of shape (3,), optional, ordered (X, Y, Z).
        Position of the first voxel in Ångströms (header words 50-52).
        Default 0.
    start : array-like of int, shape (3,), optional, ordered (X, Y, Z).
        Index of the first voxel in a parent grid (nxstart, nystart,
        nzstart; header words 5-7).  Default 0.  Some programs add it
        to the origin: set one of the two.
    """
    hdr = _header(data.shape,data.dtype,apix,ispg,origin,start)
    if fill_statistics:
        st = _statistics(data)
        hdr[19:22] = st[:3].view(_np.uint32)
        hdr[54]    = st[3:].view(_np.uint32)[0]

    with open(filename,'wb') as f:
        hdr.tofile(f)
        if data.dtype == _np.float64:
            data.astype(_np.float32).tofile(f)
        else:
            _np.ascontiguousarray(data).tofile(f)

def set_statistics(filename, data=None, mask=None, stats=None):
    """Write min, max, mean and std into the header of an existing MRC file.

    Parameters
    ----------
    filename : str
        MRC file to update (only the header statistics fields are written).
    data : numpy.ndarray, optional
        Values the statistics are computed from (normally the file's data;
        finite values only).
    mask : numpy.ndarray of bool, optional
        Voxels of ``data`` to include (e.g. the valid region of a partially
        filled volume).  Default: all.  An empty mask writes zeros.
    stats : array-like of 4, optional
        Precomputed (min, max, mean, std), instead of ``data``; e.g. from
        :class:`susan.utils.RunningStats` over chunks written with
        :func:`write_slices`.
    """
    if (data is None) == (stats is None):
        raise ValueError('Set either data or stats.')
    st = _statistics(data if mask is None else data[mask]) if stats is None else _np.array(stats,dtype=_np.float32)
    with open(filename,'r+b') as fp:
        fp.seek(76)                     # words 19-21: min, max, mean
        st[:3].tofile(fp)
        fp.seek(216)                    # word 54: rms
        st[3:4].tofile(fp)

def _layout(filename):
    """(nz, ny, nx), dtype and the byte offset of the data."""
    hdr = _np.fromfile(filename,dtype=_np.uint32,count=24)
    return (int(hdr[2]),int(hdr[1]),int(hdr[0])), _np.dtype(_mode_to_type(hdr[3])), 1024 + int(hdr[23])

def create(filename, shape, dtype=_np.float32, apix=1, ispg=None, origin=None, start=None):
    """Create an MRC2014 file of the given shape, to be filled with :func:`write_slices`.

    The header holds the final shape, pixel size and origin from the start;
    the data is allocated (sparse where the filesystem allows) and reads as
    zeros until written.  Header statistics are zero: set them with
    :func:`set_statistics` once the data is written.

    Parameters
    ----------
    filename : str
        Output file path (overwritten if it exists).
    shape : tuple of int
        (Z, Y, X).
    dtype : numpy dtype, optional
        As :func:`write` (float64 is stored as float32).  Default float32.
    apix, ispg, origin, start
        As :func:`write`.
    """
    dtype = _np.dtype(_np.float32 if _np.dtype(dtype) == _np.float64 else dtype)
    shape = tuple(int(n) for n in shape)
    hdr = _header(shape,dtype,apix,ispg,origin,start)
    with open(filename,'wb') as f:
        hdr.tofile(f)
        f.truncate(1024 + int(_np.prod(shape))*dtype.itemsize)

def write_slices(filename, data, z0):
    """Write Z-sections into an existing MRC file, starting at section ``z0``.

    Parameters
    ----------
    filename : str
        File created by :func:`create` (or any MRC file of matching shape).
    data : numpy.ndarray
        (n, Y, X) or a single (Y, X) section, with the file's dtype
        (float64 is converted to float32 for float32 files).
    z0 : int
        First section written; ``z0 + n`` must not exceed the file's Z.
    """
    (nz,ny,nx), dtype, off = _layout(filename)
    data = _np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    if data.shape[1:] != (ny,nx):
        raise ValueError('sections of %s, the file has (%d, %d)' % (data.shape[1:],ny,nx))
    if not 0 <= z0 <= nz - data.shape[0]:
        raise IndexError('sections %d..%d outside the file (%d)' % (z0,z0+data.shape[0],nz))
    if data.dtype != dtype:
        if not (data.dtype == _np.float64 and dtype == _np.float32):
            raise ValueError('data %s, the file holds %s' % (data.dtype,dtype))
        data = data.astype(_np.float32)
    with open(filename,'r+b') as f:
        f.seek(off + z0*ny*nx*dtype.itemsize)
        _np.ascontiguousarray(data).tofile(f)

def read_slices(filename, z0, z1=None, out=None):
    """Read Z-sections ``z0 .. z1-1`` of an MRC file without loading the rest.

    Parameters
    ----------
    filename : str
        Path to the MRC file.
    z0 : int
        First section.
    z1 : int, optional
        One past the last section.  Default ``z0 + 1``.
    out : numpy.ndarray, optional
        Preallocated C-contiguous (z1 - z0, Y, X) array of the file's dtype,
        filled and returned (for reading many chunks in a loop).

    Returns
    -------
    data : numpy.ndarray
        (z1 - z0, Y, X), dtype from the MRC mode (``out`` when given).
    pix_size : numpy.ndarray
        Pixel size in Ångströms, shape (3,) float32, ordered (X, Y, Z).
    """
    (nz,ny,nx), dtype, off = _layout(filename)
    z1 = z0 + 1 if z1 is None else z1
    if not 0 <= z0 < z1 <= nz:
        raise IndexError('sections %d..%d outside the file (%d)' % (z0,z1,nz))
    shape = (z1-z0,ny,nx)
    if out is None:
        out = _np.empty(shape,dtype=dtype)
    elif out.shape != shape or out.dtype != dtype or not out.flags.c_contiguous:
        raise ValueError('out must be C-contiguous %s %s' % (dtype,shape))
    with open(filename,'rb') as f:
        f.seek(off + z0*ny*nx*dtype.itemsize)
        if f.readinto(out) != out.nbytes:
            raise IOError('%s: file shorter than its header says' % filename)
    return out, get_info(filename)[1]
