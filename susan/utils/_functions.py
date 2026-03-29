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

__all__ = ['dose_from_fsc',
           'radial_average',
           'radial_expansion',
           'fsc_get',
           'fsc_analyse',
           'bandpass',
           'apply_FOM',
           'euDYN_rotm',
           'euZYZ_rotm',
           'rotm_euZYZ',
           'get_extension',
           'is_extension',
           'force_extension',
           'time_now',
           'create_sphere',
           'bin_vol',
           'mask_diameter',
           'angular_step_from_fsc',
          ]

import datetime
import susan.io.mrc as mrc
import numpy as np
from numba import jit
from os.path import splitext as split_ext
import susan.utils.datatypes as datatypes

###########################################

@jit(nopython=True,cache=True)
def radial_average(v):
    """Compute the radial (shell) average of a 3-D volume.

    Each output bin k contains the mean of all voxels at radius r ≈ k pixels
    from the volume centre.  The output length is set by the largest dimension
    so that all voxels are included.

    Parameters
    ----------
    v : ndarray, shape (Z, Y, X)
        Input 3-D volume.

    Returns
    -------
    ndarray, shape (N,)
        Radially averaged values.  N = max(Z, Y, X) // 2 + 1.
    """
    assert v.ndim == 3, "Volume must be three-dimensional"

    N = (max( max(v.shape[0],v.shape[1]), v.shape[2] )//2)+1
    val = np.zeros(N)
    wgt = np.zeros(N)

    cnt_z = v.shape[0]//2
    cnt_y = v.shape[1]//2
    cnt_x = v.shape[2]//2

    for k in range(v.shape[0]):
        z = k - cnt_z
        for j in range(v.shape[1]):
            y = j - cnt_y
            for i in range(v.shape[2]):
                x = i - cnt_x
                r = np.sqrt( x**2 + y**2 + z**2 )
                r = np.int32(np.round(r))
                if r < N:
                    val[r] += v[k,j,i]
                    wgt[r] += 1.0
    val = val/np.maximum(wgt,1)
    return val

@jit(nopython=True,cache=True)
def radial_expansion(arr):
    """Expand a 1-D radial profile into a 3-D spherically symmetric volume.

    The inverse of ``radial_average``: each voxel at radius r is assigned the
    value ``arr[round(r)]``.  The output cube has side 2*(len(arr)-1).

    Parameters
    ----------
    arr : ndarray, shape (N,)
        1-D radial profile.

    Returns
    -------
    ndarray, shape (2*(N-1), 2*(N-1), 2*(N-1)), float32
        3-D volume with spherical symmetry.
    """
    cnt = arr.shape[0]-1
    N = 2*cnt
    rslt = np.zeros((N,N,N),np.float32)

    for k in range(N):
        z = k - cnt
        for j in range(N):
            y = j - cnt
            for i in range(N):
                x = i - cnt
                r = np.sqrt( x**2 + y**2 + z**2 )
                r = np.int32(np.round(r))
                if r < cnt:
                    rslt[k,j,i] = arr[r]
    return rslt

###########################################

@jit(nopython=True,cache=True)
def _core_apply_fourier_rad_wgt(v_fou,wgt):
    c_z = v_fou.shape[0]/2
    c_y = v_fou.shape[1]/2
    for z in range(v_fou.shape[0]):
        Z = (z-c_z)**2
        for y in range(v_fou.shape[1]):
            Y = (y-c_y)**2
            for x in range(v_fou.shape[2]):
                X = x**2
                r = int(np.sqrt(X+Y+Z))
                r = min(r,wgt.shape[0]-1)
                v_fou[z,y,x] = wgt[r]*v_fou[z,y,x]

def _apply_fourier_rad_wgt(v,wgt):
    v_f = np.fft.fftshift(np.fft.rfftn(v,norm='ortho'),axes=(0,1))
    _core_apply_fourier_rad_wgt(v_f,wgt)
    rslt = np.fft.irfftn(np.fft.ifftshift(v_f,axes=(0,1)),norm='ortho')
    rslt = np.float32(rslt)
    return rslt

def _gen_bandpass_wgt(box_size,lowpass,highpass=0,rolloff=1):
    t = np.arange(box_size//2+1)
    wgt = np.ones(t.shape,np.float32)
    
    rolloff = max(rolloff,1)
    if lowpass > 0:
        x = (t-lowpass)/rolloff
        x = np.pi*x.clip(0,1)
        m = 0.5*np.cos(x)+0.5
        wgt = wgt*m
    if highpass > 0:
        x = (highpass-t)/rolloff
        x = np.pi*x.clip(0,1)
        m = 0.5*np.cos(x)+0.5
        wgt = wgt*m
    return wgt

def bandpass(v,lowpass,highpass=0,rolloff=1):
    """Apply a bandpass filter to a 3-D volume in Fourier space.

    Both the low-pass and high-pass edges use a cosine rolloff, giving a
    smooth (Hann-like) transition rather than a hard cut.

    Parameters
    ----------
    v        : ndarray, shape (Z, Y, X)
        Input volume.
    lowpass  : float
        Low-pass cutoff in Fourier pixels (0 = no low-pass).  Shells above
        this radius are attenuated.
    highpass : float, optional
        High-pass cutoff in Fourier pixels (0 = no high-pass, default).
        Shells below this radius are attenuated.
    rolloff  : int, optional
        Width of the cosine rolloff in Fourier pixels.  Default 1.

    Returns
    -------
    ndarray, float32
        Filtered volume, same shape as ``v``.
    """
    bp  = _gen_bandpass_wgt(v.shape[1],lowpass,highpass,rolloff)
    return _apply_fourier_rad_wgt(v,bp)

def apply_FOM(v,fsc_array):
    """Apply a figure-of-merit (FOM) filter derived from an FSC curve.

    Multiplies each Fourier shell by :math:`\\sqrt{FSC}`:

    .. math::
        v_{\\text{FOM}} = \\mathcal{F}^{-1}\\!\\left\\{
            \\mathcal{F}\\{v\\} \\cdot \\sqrt{\\text{fsc\\_array}}
        \\right\\}

    See `Rosenthal & Henderson (2003)
    <https://www.sciencedirect.com/science/article/pii/S104784771200144X>`_.

    Parameters
    ----------
    v         : ndarray, shape (Z, Y, X)
        Input volume.
    fsc_array : array_like, shape (N,)
        FSC curve as returned by ``fsc_get``.  Values are clipped to [0, 1]
        before taking the square root.

    Returns
    -------
    ndarray, float32
        FOM-weighted volume, same shape as ``v``.
    """
    wgt = np.sqrt(fsc_array.clip(0,1))
    return _apply_fourier_rad_wgt(v,wgt)

###########################################

@jit(nopython=True,cache=True)
def _fsc_get_core(n,d1,d2):
    rslt = np.zeros(n.shape[2],dtype=n.dtype)
    tmp1 = np.zeros(n.shape[2],dtype=n.dtype)
    tmp2 = np.zeros(n.shape[2],dtype=n.dtype)

    for k in range(n.shape[0]):
        z = k - n.shape[0]//2
        for j in range(n.shape[1]):
            y = j - n.shape[1]//2
            for i in range(n.shape[2]):
                r = np.sqrt( i**2 + y**2 + z**2 )
                r = np.int32(np.round(r))
                if r < rslt.size:
                    rslt[r] = rslt[r] +  n[k,j,i]
                    tmp1[r] = tmp1[r] + d1[k,j,i]
                    tmp2[r] = tmp2[r] + d2[k,j,i]
    rslt[0] = 1.0
    for r in range(1,rslt.size):
        denom = np.sqrt(tmp1[r]*tmp2[r])
        if denom > 0:
            rslt[r] = rslt[r] / denom
        else:
            rslt[r] = 0.0
    return rslt

def fsc_get(v1,v2,msk=None):
    """Compute the Fourier Shell Correlation (FSC) between two half-maps.

    .. math::
        FSC(r) = \\frac{
            \\text{RadialAvg}_r\\!\\left(
                \\mathcal{F}\\{v_1 \\cdot m\\} \\cdot
                \\overline{\\mathcal{F}\\{v_2 \\cdot m\\}}
            \\right)
        }{\\sqrt{
            \\text{RadialAvg}_r\\!\\left(|\\mathcal{F}\\{v_1 \\cdot m\\}|^2\\right)
            \\cdot
            \\text{RadialAvg}_r\\!\\left(|\\mathcal{F}\\{v_2 \\cdot m\\}|^2\\right)
        }}

    where *m* is the mask (1 everywhere if not provided).

    Parameters
    ----------
    v1, v2 : ndarray or str
        Input half-maps.  Can be 3-D numpy arrays or paths to MRC files.
        Both must have the same shape.
    msk    : ndarray or str or None, optional
        Real-space mask applied to both half-maps before the FFT.  Can be
        a numpy array or a path to an MRC file.  None (default) uses no mask.

    Returns
    -------
    ndarray, shape (N,)
        FSC curve.  Shell 0 is set to 1.0; N = v1.shape[2] // 2 + 1.
    """
    apix = 1
    if isinstance(v1,str):
        v1,apix = mrc.read(v1)
    
    if isinstance(v2,str):
        v2,_ = mrc.read(v2)
    
    if msk is not None:
        if isinstance(msk,str):
            msk,_ = mrc.read(msk)
        
        v1 = v1*msk
        v2 = v2*msk

    V1 = np.fft.fftshift( np.fft.rfftn(v1,norm='ortho'), axes=(0,1))
    V2 = np.fft.fftshift( np.fft.rfftn(v2,norm='ortho'), axes=(0,1))
    
    num = np.real(V1*np.conjugate(V2)).astype(np.float32)
    d_1 = np.real(V1*np.conjugate(V1)).astype(np.float32)
    d_2 = np.real(V2*np.conjugate(V2)).astype(np.float32)
    
    fsc = _fsc_get_core(num,d_1,d_2)
    
    return fsc

def fsc_analyse(fsc,apix=1.0,thres=0.143):
    """Find the resolution where the FSC drops below a threshold.

    Parameters
    ----------
    fsc   : array_like
        FSC curve as returned by ``fsc_get``.
    apix  : float or array_like, optional
        Pixel size in Angstroms.  Default 1.0 (returns resolution in pixels).
    thres : float, optional
        FSC threshold.  Default 0.143 (gold-standard half-map criterion).

    Returns
    -------
    datatypes.fsc_info
        Named tuple with fields:

        * ``fpix``  — resolution in Fourier pixels (int).
        * ``res``   — resolution in Angstroms (float).  0.0 if the FSC never
          drops below ``thres``.
    """
    apix = np.array(apix)
    if( apix.size > 1 ):
        apix = apix[0]
    fpix = np.argwhere(fsc<thres)
    if fpix.size > 0:
        fpix = fpix[0,0]
    else:
        fpix = fsc.size-1
    if fpix == 0:
        res = 0
    else:
        res  = (2*(fsc.size-1)*apix)/fpix
    rslt = datatypes.fsc_info(fpix,res)
    return rslt

###########################################

@jit(nopython=True,cache=True)
def euDYN_rotm(R,eu):
    """Fill rotation matrix R from dynamic (intrinsic ZXZ-like) Euler angles.

    Uses SUSAN's internal DYN convention:
    eu = [psi, phi, theta] in radians, where phi is the tilt (polar) angle
    measured from the Z-axis.  The rotation is applied in the order
    theta → phi → psi (intrinsic, right-handed).

    Parameters
    ----------
    R  : ndarray, shape (3, 3)
        Output array filled in-place.
    eu : ndarray, shape (3,)
        Euler angles [psi, phi, theta] in radians.
    """
    cos_theta = np.cos(eu[2])
    cos_phi   = np.cos(eu[1])
    cos_psi   = np.cos(eu[0])
    sin_theta = np.sin(eu[2])
    sin_phi   = np.sin(eu[1])
    sin_psi   = np.sin(eu[0])
    R[0,0] = cos_theta*cos_psi - cos_phi*sin_theta*sin_psi
    R[0,1] = -cos_theta*sin_psi - cos_phi*cos_psi*sin_theta
    R[0,2] = sin_theta*sin_phi
    R[1,0] = cos_psi*sin_theta + cos_theta*cos_phi*sin_psi
    R[1,1] = cos_theta*cos_phi*cos_psi - sin_theta*sin_psi
    R[1,2] = -cos_theta*sin_phi
    R[2,0] = sin_phi*sin_psi
    R[2,1] = cos_psi*sin_phi
    R[2,2] = cos_phi

@jit(nopython=True,cache=True)
def euZYZ_rotm(R,eu):
    """Fill rotation matrix R from ZYZ Euler angles.

    Standard extrinsic ZYZ convention:
    R = R_z(theta) · R_y(phi) · R_z(psi).

    Parameters
    ----------
    R  : ndarray, shape (3, 3)
        Output array filled in-place.
    eu : ndarray, shape (3,)
        Euler angles [theta, phi, psi] in radians.
    """
    cos_theta = np.cos(eu[0])
    cos_phi = np.cos(eu[1])
    cos_psi = np.cos(eu[2])
    sin_theta = np.sin(eu[0])
    sin_phi = np.sin(eu[1])
    sin_psi = np.sin(eu[2])
    R[0,0] = cos_theta*cos_phi*cos_psi - sin_theta*sin_psi
    R[0,1] = -cos_psi*sin_theta - cos_theta*cos_phi*sin_psi
    R[0,2] = cos_theta*sin_phi
    R[1,0] = cos_theta*sin_psi + cos_phi*cos_psi*sin_theta
    R[1,1] = cos_theta*cos_psi - cos_phi*sin_theta*sin_psi
    R[1,2] = sin_theta*sin_phi
    R[2,0] = -cos_psi*sin_phi
    R[2,1] = sin_phi*sin_psi
    R[2,2] = cos_phi

@jit(nopython=True,cache=True)
def rotm_euZYZ(euler, R):
    """Extract ZYZ Euler angles from a rotation matrix.

    Inverse of ``euZYZ_rotm``.  Handles the two degenerate cases
    (phi=0 and phi=π) by setting theta=0 and expressing the remaining
    degree of freedom in psi.

    Parameters
    ----------
    euler : ndarray, shape (3,)
        Output array filled in-place with [theta, phi, psi] in radians.
    R     : ndarray, shape (3, 3)
        Input rotation matrix.
    """
    if np.abs(R[2,2]-1)<1e-6:
        # phi=0: only (theta+psi) is determined; set theta=0
        euler[0] = 0
        euler[1] = 0
        euler[2] = np.arctan2(R[1,0], R[1,1])
    elif np.abs(R[2,2]+1)<1e-6:
        # phi=pi: only (psi-theta) is determined; set theta=0
        euler[0] = 0
        euler[1] = np.pi
        euler[2] = np.arctan2(R[1,0], R[1,1])
    else:
        euler[0] = np.arctan2(R[1,2],R[0,2])
        euler[1] = np.arctan2(np.sqrt(np.abs(1-(R[2,2]*R[2,2]))),R[2,2])
        euler[2] = np.arctan2(R[2,1],-R[2,0])

###########################################

def get_extension(filename):
    """Return the file extension including the leading dot.

    Parameters
    ----------
    filename : str

    Returns
    -------
    str
        Extension, e.g. ``'.mrc'``.  Empty string if there is no extension.
    """
    _,ext = split_ext(filename)
    return ext

def is_extension(filename,extension):
    """Check whether ``filename`` has the given extension (case-sensitive).

    Parameters
    ----------
    filename  : str
    extension : str
        With or without a leading dot (both forms are accepted).

    Returns
    -------
    bool
    """
    _,ext = split_ext(filename)
    if( extension[0] == '.' ):
        return ext == extension
    else:
        return ext == '.'+extension

def force_extension(filename,extension):
    """Return ``filename`` with its extension replaced by ``extension``.

    Parameters
    ----------
    filename  : str
    extension : str
        With or without a leading dot (both forms are accepted).

    Returns
    -------
    str
        Path with the new extension.
    """
    base,ext = split_ext(filename)
    new_ext = extension
    if new_ext[0] != '.':
        new_ext = '.' + extension
    return base + new_ext

###########################################

def time_now():
    """Return the current local date and time.

    Returns
    -------
    datetime.datetime
    """
    return datetime.datetime.now()

###########################################

def create_sphere(r,N):
    """Create a soft spherical mask of radius ``r`` in a cube of side ``N``.

    The mask value at each voxel is ``clip(r - radius, 0, 1)``, giving a
    smooth 1-pixel-wide transition at the sphere boundary.

    Parameters
    ----------
    r : float
        Sphere radius in pixels.
    N : int
        Side length of the output cube.

    Returns
    -------
    ndarray, shape (N, N, N), float32
        Soft spherical mask; 1 inside, 0 outside, linear transition at edge.
    """
    M = N//2
    t = np.arange(-M,M)
    x,y,z = np.meshgrid(t,t,t)
    rad = np.sqrt( x**2 + y**2 + z**2 )
    return np.float32((r-rad).clip(0,1))

###########################################

def bin_vol(vol,bin_level):
    """Low-pass filter and downsample a volume by a power of two.

    Applies a low-pass filter at the new Nyquist frequency before
    downsampling to prevent aliasing.

    Parameters
    ----------
    vol       : ndarray, shape (N, N, N)
        Input volume.
    bin_level : int
        Downsampling factor as a power of two.  bin_level=1 halves each
        dimension; bin_level=2 quarters it, etc.

    Returns
    -------
    ndarray, float32
        Downsampled volume of shape (N//s, N//s, N//s) where s = 2**bin_level.
    """
    s = (2**bin_level)
    v = bandpass(vol,vol.shape[0]//(2*s)-1)
    v = v[::s,::s,::s]
    return np.float32(v)

###########################################

def mask_diameter(mask_file, threshold=0.5):
    """Estimate the particle diameter in pixels from a soft mask MRC file.

    The diameter is that of the sphere whose volume equals the volume of mask
    voxels above *threshold*.  Returning pixels (not Angstroms).

    Parameters
    ----------
    mask_file : str
        Path to the mask MRC file.
    threshold : float, optional
        Voxel values above this level are considered 'inside' the mask.
        Default 0.5 works for all standard soft masks.

    Returns
    -------
    float
        Equivalent-sphere diameter in pixels.
    """
    mask, _ = mrc.read(mask_file)
    n_inside    = float(np.sum(mask > threshold))
    # V_pix = n_inside voxels  →  D_pix = 2·(3·V/(4π))^(1/3)
    diameter_px = 2.0 * (3.0 * n_inside / (4.0 * np.pi)) ** (1.0 / 3.0)
    return diameter_px

###########################################

def angular_step_from_fsc(fsc_fpix):
    """Angular step from an FSC resolution in Fourier pixels.

    Returns the angle subtended by one Fourier pixel at the resolution shell
    ``fsc_fpix``::

        Δθ = atan2(1, fsc_fpix)   [degrees]

    This is the smallest orientation change that moves the projected signal
    by one pixel at the resolution limit — i.e. the Nyquist angular step for
    the given resolution.  No pixel size or particle diameter is needed.

    Parameters
    ----------
    fsc_fpix : int or float
        Resolution in Fourier pixels as returned by ``fsc_analyse``.

    Returns
    -------
    float
        Suggested angular step in degrees.
    """
    if fsc_fpix <= 0:
        return float('inf')
    return float(np.degrees(np.arctan2(1.0, float(fsc_fpix))))

###########################################

def dose_from_fsc(fsc, apix, freq_range=(0.1, 0.8), fsc_min=0.1):
    """Estimate effective dose from the Guinier slope of the FSC curve.

    The ExpFilt dose is applied in reconstruction as exp(−s²·dose/4), where s
    is in 1/Å.  In the intermediate frequency range the FSC decays as the same
    Gaussian envelope, so fitting ln(FSC) vs s² gives slope = −dose/4, and:

        dose = −4 · d(ln FSC)/d(s²)

    This can be compared to the mean of ``ptcls.def_ExFl`` (excluding failures
    marked as 9999) to calibrate ``aligner.expfilt_gain``:

        expfilt_gain = dose_from_fsc(fsc, apix) / mean_estimated_dose

    Parameters
    ----------
    fsc : array_like
        FSC curve as returned by ``fsc_get``.  Assumed to have n shells
        spanning a box of size 2n (i.e. shell k → s = k / (2n·apix)).
    apix : float
        Pixel size in Angstroms.
    freq_range : tuple of float
        (low, high) as fractions of Nyquist over which to fit.  The default
        (0.1, 0.8) covers the Guinier decay while stopping before the
        noise-dominated tail.
    fsc_min : float
        Minimum FSC value included in the fit.  Shells at or below the noise
        floor would bias the slope.  Default 0.1.

    Returns
    -------
    float
        Effective dose in Å² consistent with the ExpFilt convention.
        Returns NaN if the fit cannot be performed.
    """
    fsc   = np.asarray(fsc, dtype=np.float64)
    n     = len(fsc)
    s_nyq = 1.0 / (2.0 * float(apix))
    s     = np.arange(n) / n * s_nyq   # shell k → s = k/(2n·apix); s[n-1] ≈ s_nyq
    s2    = s * s

    lo, hi = freq_range
    mask   = (s >= lo * s_nyq) & (s <= hi * s_nyq) & (fsc > fsc_min)
    if mask.sum() < 3:
        return float('nan')

    slope, _ = np.polyfit(s2[mask], np.log(fsc[mask]), 1)
    return -4.0 * slope   # dose = −4 · slope  (matches exp(−s²·dose/4) convention)
