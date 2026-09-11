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
           'fsc_get_fpix',
           'ssnr_from_fsc',
           'bandpass',
           'apply_FOM',
           'fsc_sharpen',
           'fsc_sharpen_filter',
           'euDYN_rotm',
           'euZYZ_rotm',
           'rotm_euZYZ',
           'get_extension',
           'is_extension',
           'force_extension',
           'time_now',
           'create_sphere',
           'bin_vol',
           'bin_frame',
           'bin_frame_shape',
           'mask_diameter',
           'angular_step_from_fsc',
           'is_odd',
           'is_even',
          ]

import datetime
import warnings as _warnings
import susan.io.mrc as mrc
import numpy as np

def _warn(msg):
    _warnings.warn(msg,RuntimeWarning,stacklevel=3)

from os.path import splitext as split_ext
from susan.utils._functions_core import (
    radial_average as _radial_average_cy,
    radial_expansion,
    _core_apply_fourier_rad_wgt,
    _fsc_get_core,
    euDYN_rotm,
    euZYZ_rotm,
    rotm_euZYZ,
    bin_frame as _bin_frame_cy,
    bin_frame_shape as _bin_frame_shape_cy,
)
import susan.utils.datatypes as datatypes

###########################################

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
    return _radial_average_cy(np.ascontiguousarray(v, dtype=np.float64))

###########################################


def _apply_fourier_rad_wgt(v,wgt):
    v_f = np.ascontiguousarray(np.fft.fftshift(np.fft.rfftn(v.astype(float),norm='ortho'),axes=(0,1)))
    _core_apply_fourier_rad_wgt(v_f, np.ascontiguousarray(wgt, dtype=np.float32))
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


def fsc_sharpen(v, fsc, apix, bfactor, fom='rosenthal',
                lowpass=True, thres=0.143, rolloff=2):
    """FSC-weighted B-factor sharpening of a map from its half-map FSC.

    Builds a single radial Fourier weight combining up to three per-shell
    terms and applies it to *v*:

    1. **B-factor amplification** ``exp(-bfactor * s^2 / 4)`` where
       ``s = k / (N * apix)`` is the spatial frequency (1/Angstrom) of shell
       ``k``.  A **negative** *bfactor* sharpens (boosts high frequencies); a
       positive one blurs.
    2. **FOM weighting** derived from the FSC, which tapers the amplification
       to zero where the half-maps stop correlating, so noise beyond the
       resolution limit is not amplified.  ``'rosenthal'`` uses the full-map
       figure of merit :math:`\\sqrt{2\\,FSC/(1+FSC)}` (appropriate for the
       combined map; Rosenthal & Henderson, 2003); ``'sqrt'`` uses
       :math:`\\sqrt{FSC}` (matches :func:`apply_FOM`); ``None`` disables it.
    3. **Cosine low-pass** at the FSC resolution (:func:`fsc_analyse` with
       *thres*), a safety cut so shells past the resolution are not boosted.

    This is the principled alternative to a blind (FSC-agnostic) B-factor:
    the amplification is gated by where there is real, reproducible signal.

    Parameters
    ----------
    v : ndarray, shape (Z, Y, X)
        Map to sharpen (typically the combined reconstruction).
    fsc : array_like, shape (N//2+1,)
        Half-map FSC curve, as returned by :func:`fsc_get`.
    apix : float
        Pixel size in Angstroms.
    bfactor : float
        B-factor in Angstrom^2.  Negative sharpens, positive blurs.
    fom : {'rosenthal', 'sqrt', None}, optional
        FSC figure-of-merit weighting.  Default ``'rosenthal'``.
    lowpass : bool, optional
        Apply a cosine low-pass at the FSC resolution.  Default ``True``.
    thres : float, optional
        FSC threshold used to locate the low-pass edge.  Default ``0.143``.
    rolloff : int, optional
        Cosine rolloff width (Fourier pixels) of the low-pass.  Default ``2``.

    Returns
    -------
    ndarray, float32
        Sharpened volume, same shape as *v*.

    See Also
    --------
    apply_FOM : FSC figure-of-merit weighting only (no B-factor).
    fsc_sharpen_filter : wrap this as a ``map_filter_fsc`` callable for
        :class:`~susan.project.SubtomoAvg.SubtomoAvg`.
    """
    apix = float(np.asarray(apix).flatten()[0])
    fsc  = np.asarray(fsc, dtype=np.float64)
    N    = v.shape[-1]
    k    = np.arange(N // 2 + 1)
    if fsc.size != k.size:
        raise ValueError('fsc length (%d) does not match box size (expected %d)'
                         % (fsc.size, k.size))

    s   = k / (N * apix)                       # spatial frequency [1/A]
    wgt = np.exp(-bfactor * s * s / 4.0)       # bfactor < 0 -> amplify

    if fom is not None:
        f = fsc.clip(0, 1)
        if fom == 'rosenthal':
            c = np.sqrt(np.clip(2 * f / (1 + f), 0, 1))
        elif fom == 'sqrt':
            c = np.sqrt(f)
        else:
            raise ValueError("fom must be 'rosenthal', 'sqrt', or None")
        wgt = wgt * c

    if lowpass:
        fpix = fsc_analyse(fsc, apix, thres).fpix
        wgt  = wgt * _gen_bandpass_wgt(N, fpix, 0, rolloff)

    return _apply_fourier_rad_wgt(v, wgt.astype(np.float32))


def fsc_sharpen_filter(apix, bfactor, **kwargs):
    """Build a ``map_filter_fsc`` callable that applies :func:`fsc_sharpen`.

    The returned function has signature ``filter(vol, fsc) -> vol``, matching
    :attr:`SubtomoAvg.map_filter_fsc <susan.project.SubtomoAvg.SubtomoAvg>`, so
    it plugs directly into the post-processing hook — the project passes the
    per-reference FSC in automatically each iteration.

    Parameters
    ----------
    apix : float
        Pixel size in Angstroms (e.g. ``sta.pix_size``).
    bfactor : float
        B-factor in Angstrom^2 (negative sharpens).
    **kwargs
        Forwarded to :func:`fsc_sharpen` (``fom``, ``lowpass``, ``thres``,
        ``rolloff``).

    Returns
    -------
    callable
        ``lambda vol, fsc: fsc_sharpen(vol, fsc, apix, bfactor, **kwargs)``.

    Examples
    --------
    >>> sta.map_filter_fsc = fsc_sharpen_filter(sta.pix_size, bfactor=-120)
    """
    return lambda vol, fsc: fsc_sharpen(vol, fsc, apix, bfactor, **kwargs)

###########################################


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
    
    num = np.ascontiguousarray(np.real(V1*np.conjugate(V2)), dtype=np.float32)
    d_1 = np.ascontiguousarray(np.real(V1*np.conjugate(V1)), dtype=np.float32)
    d_2 = np.ascontiguousarray(np.real(V2*np.conjugate(V2)), dtype=np.float32)
    
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

def fsc_get_fpix(fsc,th_list=(0.5,0.143),interp=True):
    """Resolution in Fourier pixels at each of several FSC thresholds.

    Unlike :func:`fsc_analyse`, this always returns a list, one entry per
    threshold, and can interpolate the crossing to sub-shell precision.

    Parameters
    ----------
    fsc : array_like
        FSC curve as returned by :func:`fsc_get` (``n = box//2 + 1`` shells).
    th_list : float or sequence of float, optional
        FSC threshold(s).  A scalar is accepted and treated as a 1-element
        list.  Default ``(0.5, 0.143)``.
    interp : bool, optional
        If True (default) linearly interpolate between the two shells
        bracketing the crossing.  Matters when the crossings are only a few
        shells apart, which is typical of CryoET half-map FSCs: with integer
        crossings the anchors used by :func:`ssnr_from_fsc` can be off by
        more than 15%.

    Returns
    -------
    list of float
        One entry per threshold, always a list even for a single threshold.

        Sentinels:

        * ``nan`` — the FSC never drops below the threshold.
        * ``0.0`` — the FSC is already below the threshold at shell 0.
    """
    fsc     = np.asarray(fsc,dtype=np.float64)
    th_list = np.atleast_1d(np.asarray(th_list,dtype=np.float64))

    rslt = []
    for th in th_list:
        below = fsc < th
        if not below.any():
            rslt.append(float('nan'))
            continue
        i = int(np.argmax(below))
        if i == 0:
            rslt.append(0.0)
            continue
        if interp:
            f0,f1 = fsc[i-1],fsc[i]
            frac  = (f0-th)/(f0-f1) if f0 > f1 else 0.0
            rslt.append(float(i-1+frac))
        else:
            rslt.append(float(i))
    return rslt

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

def create_sphere(r,N,center=None):
    """Create a soft spherical mask of radius ``r`` in a cube of side ``N``.

    The mask value at each voxel is ``clip(r - radius, 0, 1)``, giving a
    smooth 1-pixel-wide transition at the sphere boundary.

    Parameters
    ----------
    r : float
        Sphere radius in pixels.
    N : int
        Side length of the output cube.
    center : array-like of 3 floats, optional
        Voxel coordinates ``(z, y, x)`` of the sphere centre.  ``None``
        (default) places it at the geometric centre ``(N//2, N//2, N//2)``,
        reproducing the original behaviour.  Use this to place a mask at an
        arbitrary location (e.g. a tile centre in local processing).

    Returns
    -------
    ndarray, shape (N, N, N), float32
        Soft spherical mask; 1 inside, 0 outside, linear transition at edge.
    """
    M = N//2
    if center is None:
        center = (M, M, M)
    a0 = np.arange(N) - center[0]
    a1 = np.arange(N) - center[1]
    a2 = np.arange(N) - center[2]
    x0, x1, x2 = np.meshgrid(a0, a1, a2, indexing='ij')
    rad = np.sqrt( x0**2 + x1**2 + x2**2 )
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

def bin_frame_shape(H, W, scale):
    """Return the (H_b, W_b) shape that :func:`bin_frame` would produce."""
    return _bin_frame_shape_cy(int(H), int(W), float(scale))

def bin_frame(in_frame, scale, out_frame=None):
    """Area-weighted downsample of a single 2-D frame by a float ``scale``.

    Output dimensions are ``ceil(H/scale)`` and ``ceil(W/scale)``.  The window
    offset ``(N - N_b*scale)/2 - (scale-1)/2`` keeps the sampling origin on
    SUSAN's pixel-centre convention (input index ``i`` at coordinate ``i``,
    tomogram centre at ``stk_center = N/2``), so the same particle position
    projects to the same physical point across binning levels.  Preserving the
    geometric box-edge centre instead would shift binned content by
    ``(scale-1)/2`` input pixels and blur the reconstruction across tilts.

    Edge bins extend past the input boundary; out-of-bounds contributions are
    skipped and each output pixel is normalised by the actual in-bounds
    weight, so no artificial padding is introduced.

    Parameters
    ----------
    in_frame  : ndarray, shape (H, W)
        Input frame; converted to contiguous float32 if needed.
    scale     : float, > 1.0
        Downsampling factor (input pixels per output pixel).
    out_frame : ndarray, optional
        Pre-allocated output buffer of shape ``(ceil(H/scale), ceil(W/scale))``,
        dtype float32, contiguous.  Allocated internally if not given.

    Returns
    -------
    ndarray, float32, shape (ceil(H/scale), ceil(W/scale))
        The downsampled frame.
    """
    if scale <= 1.0:
        raise ValueError("scale must be > 1.0")
    in_frame = np.ascontiguousarray(in_frame, dtype=np.float32)
    if in_frame.ndim != 2:
        raise ValueError("in_frame must be 2-D")
    H_b, W_b = _bin_frame_shape_cy(in_frame.shape[0], in_frame.shape[1], float(scale))
    if out_frame is None:
        out_frame = np.empty((H_b, W_b), dtype=np.float32)
    elif out_frame.shape != (H_b, W_b) or out_frame.dtype != np.float32 \
         or not out_frame.flags['C_CONTIGUOUS']:
        raise ValueError(
            "out_frame must be C-contiguous float32 of shape (%d, %d)"
            % (H_b, W_b)
        )
    _bin_frame_cy(out_frame, in_frame, float(scale))
    return out_frame

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
        spanning a box of side 2(n-1), i.e. shell k → s = k / (2(n-1)·apix),
        matching ``fsc_get`` and ``fsc_analyse``.
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
    if n < 2:
        return float('nan')
    box   = 2*(n-1)
    s_nyq = 1.0 / (2.0 * float(apix))
    s     = np.arange(n) / (box * float(apix))   # shell k → s = k/(box·apix); s[n-1] = s_nyq
    s2    = s * s

    lo, hi = freq_range
    mask   = (s >= lo * s_nyq) & (s <= hi * s_nyq) & (fsc > fsc_min)
    if mask.sum() < 3:
        return float('nan')

    slope, _ = np.polyfit(s2[mask], np.log(fsc[mask]), 1)
    return -4.0 * slope   # dose = −4 · slope  (matches exp(−s²·dose/4) convention)

###########################################

def ssnr_from_fsc(fsc,apix,th_list=(0.5,0.143),n_eff=None,fallback=True):
    """Estimate the ad-hoc SSNR parameters (S, F) from an FSC curve.

    SUSAN models the spectral SNR as (see :class:`susan.utils.datatypes.ssnr`)

    .. math::
        SSNR(s) = 10^{3S} \\cdot e^{-100 F s}, \\quad s \\text{ in } 1/\\text{\\AA}

    so :math:`\\ln SSNR` is linear in *s* and two points determine it.  The
    two anchors are taken from the FSC itself, converting each threshold *t*
    to the map SSNR it corresponds to, :math:`SSNR = 2t/(1-t)`:

    .. math::
        F = \\frac{\\ln(q_1/q_2)\\,(box \\cdot apix)}{100\\,(r_2-r_1)}, \\quad
        S = \\frac{\\ln q_1 + \\ln(q_1/q_2)\\, r_1/(r_2-r_1)}{3 \\ln 10}

    with :math:`r_i` the crossing radii in Fourier pixels and
    :math:`q_i = 2t_i/(1-t_i)`.  Note that *S* depends only on the ratio
    :math:`r_1/(r_2-r_1)` and so is independent of the pixel size; *F* scales
    with ``box*apix``.

    The two-anchor solve is used rather than a least-squares fit because the
    applied weight, ``SSNR/(1+SSNR)``, saturates at 0 and 1: only the location
    and sharpness of the turnover matter, and those are what the anchors fix.

    Parameters
    ----------
    fsc : array_like
        FSC curve as returned by :func:`fsc_get` (``n = box//2 + 1`` shells,
        so the box side is ``2*(n-1)``).
    apix : float
        Pixel size in Angstroms of the maps the FSC was computed from.
        Needed for *F*; *S* does not depend on it.
    th_list : sequence of two float, optional
        The two FSC anchors, in decreasing order.  Default ``(0.5, 0.143)``.
    n_eff : float or None, optional
        Effective number of independent 2D measurements contributing to the
        map, roughly ``n_particles * n_tilts`` (times the symmetry order; use
        ``sum(prj_w)`` in place of ``n_tilts`` if the weights are not all 1).
        The FSC measures the SSNR of the *map*, while the model is defined as
        the SSNR of a *single projection*, so ``S`` is reduced by
        ``log10(n_eff)/3`` (``F`` is unchanged).  Pass it when the result is
        destined for the substack whitening or the reconstruction Wiener
        filter, both of which apply the SSNR once per projection.  ``None``
        (default) returns the map SSNR unconverted.

        Note that with ``n_eff=None`` the returned ``S`` is always positive
        (every term of its formula is, as long as ``th_list[0] > 1/3``), so
        the ``10^(3S) > 1`` gate in the C++ ``radial_frc_acc`` always engages.
        Applying the conversion can push ``S`` below 0 on a low-SSNR dataset,
        which that gate reads as "disabled" and silently reverts to pure
        whitening.
    fallback : bool, optional
        What to do when the requested anchors are not both reached.  Default
        True, which walks down this ladder:

        1. Both ``th_list`` crossings usable — solve directly.
        2. Only ``th_list[0]`` crossed — re-anchor on
           ``((1+th_list[0])/2, th_list[0])``, i.e. ``(0.75, 0.5)`` for the
           defaults.  The estimate is then extrapolated past the data, and a
           warning is issued.
        3. Neither crossed — the map is Nyquist-limited rather than
           SNR-limited.  Anchor on the curve itself (first shell below 0.99
           and the outermost shell) so the taper follows the FSC value at
           Nyquist.  It is correspondingly mild, tending to no taper at all
           as the FSC flattens.  A warning is issued.

        Set False to get ``nan`` instead of any fallback.

    Returns
    -------
    datatypes.ssnr
        Named tuple with fields ``S`` and ``F``.

        Both fields are ``nan`` if no usable pair of anchors exists: the
        crossings are out of order or coincident, the first anchor sits at
        shell 0, the curve has fewer than 2 shells, or ``fallback`` is False
        and the requested anchors were not both reached.  Check with
        ``math.isnan(rslt.F)`` before use.

    Raises
    ------
    ValueError
        If ``th_list`` does not hold exactly two decreasing values in (0, 1),
        or if ``apix`` is not positive.
    """
    fsc = np.asarray(fsc,dtype=np.float64)

    th_list = np.atleast_1d(np.asarray(th_list,dtype=np.float64))
    if th_list.size != 2:
        raise ValueError('ssnr_from_fsc needs exactly 2 thresholds (the two anchors).')
    t1,t2 = float(th_list[0]),float(th_list[1])
    if not (0.0 < t2 < t1 < 1.0):
        raise ValueError('The thresholds must satisfy 0 < th_list[1] < th_list[0] < 1.')
    if not (float(apix) > 0.0):
        raise ValueError('apix must be larger than 0.')
    if n_eff is not None and not (float(n_eff) > 0.0):
        raise ValueError('n_eff must be larger than 0 (or None to skip the conversion).')
    if fsc.size < 2:
        return datatypes.ssnr(float('nan'),float('nan'))

    def _q(t):                       # FSC threshold -> SSNR of the map
        t = min(float(t),1.0-1e-6)   # clamp: FSC == 1 would give an infinite SSNR
        return 2.0*t/(1.0-t)

    r1,r2 = fsc_get_fpix(fsc,(t1,t2),interp=True)
    q1,q2 = _q(t1),_q(t2)

    ok1 = bool(np.isfinite(r1)) and (r1 > 0.0)
    ok2 = bool(np.isfinite(r2))

    if ok1 and ok2 and (r2 > r1):
        pass                         # level 1: both requested anchors usable
    elif not fallback:
        return datatypes.ssnr(float('nan'),float('nan'))
    elif ok1:
        # Level 2: the outer anchor was never reached, but the inner one was.
        # Move both anchors up: t1 becomes the FSC halfway between 1 and t1
        # (0.75 for the default t1 = 0.5) and the old t1 becomes the outer one.
        t1b,t2b = 0.5*(1.0+t1),t1
        r1,r2   = fsc_get_fpix(fsc,(t1b,t2b),interp=True)
        q1,q2   = _q(t1b),_q(t2b)
        if (not np.isfinite(r1)) or (not np.isfinite(r2)) or (r2 <= r1):
            return datatypes.ssnr(float('nan'),float('nan'))
        _warn('ssnr_from_fsc: FSC never reaches %g; anchoring on (%g, %g) instead.'%(t2,t1b,t2b))
    else:
        # Level 3: not even the inner anchor was reached, so the map is
        # Nyquist-limited rather than SNR-limited.  Anchor on the curve
        # itself: the first shell safely below 1 and the outermost shell.
        # The taper this yields is set by the FSC value at Nyquist, and is
        # correspondingly mild (it tends to no taper as the FSC flattens).
        usable = np.argwhere(fsc < 0.99)
        if usable.size == 0:
            return datatypes.ssnr(float('nan'),float('nan'))
        r1,r2  = float(usable[0,0]),float(fsc.size-1)
        q1,q2  = _q(fsc[int(r1)]),_q(fsc[-1])
        if r2 <= r1:
            return datatypes.ssnr(float('nan'),float('nan'))
        _warn('ssnr_from_fsc: FSC never reaches %g; anchoring on the curve '
              '(shells %d and %d, FSC %.3f and %.3f). The taper will be mild.'
              %(t1,int(r1),int(r2),fsc[int(r1)],fsc[-1]))

    # q1 == q2 (a flat FSC) is fine and yields F = 0, i.e. no taper.  A
    # non-positive SSNR carries no information, and q1 < q2 would mean the
    # SSNR rises with frequency; neither is a usable anchor pair.
    if (not (q1 > 0.0)) or (not (q2 > 0.0)) or (q1 < q2):
        return datatypes.ssnr(float('nan'),float('nan'))

    lq  = np.log(q1/q2)
    box = 2*(fsc.size-1)
    F   = lq*(box*float(apix))/(100.0*(r2-r1))
    S   = (np.log(q1) + lq*r1/(r2-r1))/(3.0*np.log(10.0))

    if n_eff is not None:
        S -= np.log10(float(n_eff))/3.0

    return datatypes.ssnr(float(S),float(F))

###########################################

def is_odd(v):
    """Check whether an integer is odd.

    Parameters
    ----------
    v : int

    Returns
    -------
    bool
        True if ``v`` is odd, False otherwise.
    """
    return int(v) % 2 == 1

def is_even(v):
    """Check whether an integer is even.

    Parameters
    ----------
    v : int

    Returns
    -------
    bool
        True if ``v`` is even, False otherwise.
    """
    return int(v) % 2 == 0
