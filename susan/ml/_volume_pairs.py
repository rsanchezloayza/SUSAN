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

__all__ = ['VolumePairs']

import numpy as _np
import torch  as _torch


# ---------------------------------------------------------------------------
# Angular perturbation helpers
#
# A "proper" angular perturbation is an isotropic small rotation on SO(3):
# a uniformly random axis with a Gaussian-distributed magnitude.  We build it
# as a rotation matrix (Rodrigues), compose it with the particle rotation, and
# only then convert back to ZYZ Euler angles.  Working through matrices avoids
# the gimbal-lock artefacts of adding noise directly to Euler components.
#
# The Euler<->matrix formulas below replicate ``euZYZ_rotm`` / ``rotm_euZYZ``
# in ``susan/utils/_functions_core.pyx`` exactly, so the ZYZ convention matches
# the rest of SUSAN (ali_eu slot 0 = polar, 1 = azimuth, 2 = in-plane).


def _euZYZ_to_R(eu: _np.ndarray) -> _np.ndarray:
    """Vectorised ZYZ Euler (radians, ``(N, 3)``) to rotation matrices ``(N, 3, 3)``."""
    t, p, s = eu[:, 0], eu[:, 1], eu[:, 2]
    ct, cp, cs = _np.cos(t), _np.cos(p), _np.cos(s)
    st, sp, ss = _np.sin(t), _np.sin(p), _np.sin(s)
    R = _np.empty((eu.shape[0], 3, 3), dtype=_np.float64)
    R[:, 0, 0] =  ct*cp*cs - st*ss
    R[:, 0, 1] = -cs*st - ct*cp*ss
    R[:, 0, 2] =  ct*sp
    R[:, 1, 0] =  ct*ss + cp*cs*st
    R[:, 1, 1] =  ct*cs - cp*st*ss
    R[:, 1, 2] =  st*sp
    R[:, 2, 0] = -cs*sp
    R[:, 2, 1] =  sp*ss
    R[:, 2, 2] =  cp
    return R


def _R_to_euZYZ(R: _np.ndarray) -> _np.ndarray:
    """Vectorised rotation matrices ``(N, 3, 3)`` to ZYZ Euler (radians, ``(N, 3)``)."""
    eu  = _np.empty((R.shape[0], 3), dtype=_np.float64)
    r22 = R[:, 2, 2]
    gimbal = _np.abs(_np.abs(r22) - 1.0) < 1e-6

    # General case.
    eu[:, 0] = _np.arctan2(R[:, 1, 2], R[:, 0, 2])
    eu[:, 1] = _np.arctan2(_np.sqrt(_np.abs(1.0 - r22*r22)), r22)
    eu[:, 2] = _np.arctan2(R[:, 2, 1], -R[:, 2, 0])

    # Gimbal-lock case (R22 ~ +-1): polar collapses, only the sum/diff of the
    # two remaining angles is defined; follow rotm_euZYZ and fold it into slot 2.
    eu[gimbal, 0] = 0.0
    eu[gimbal, 1] = _np.where(r22[gimbal] > 0, 0.0, _np.pi)
    eu[gimbal, 2] = _np.arctan2(R[gimbal, 1, 0], R[gimbal, 1, 1])
    return eu


def _rotvec_to_R(rvec: _np.ndarray) -> _np.ndarray:
    """Rodrigues: rotation vectors ``(N, 3)`` (axis*angle, radians) to matrices."""
    theta = _np.linalg.norm(rvec, axis=1)
    small = theta < 1e-8
    axis  = _np.zeros_like(rvec)
    nz    = ~small
    axis[nz] = rvec[nz] / theta[nz, None]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c = _np.cos(theta); s = _np.sin(theta); C = 1.0 - c
    R = _np.empty((rvec.shape[0], 3, 3), dtype=_np.float64)
    R[:, 0, 0] = c + x*x*C
    R[:, 0, 1] = x*y*C - z*s
    R[:, 0, 2] = x*z*C + y*s
    R[:, 1, 0] = y*x*C + z*s
    R[:, 1, 1] = c + y*y*C
    R[:, 1, 2] = y*z*C - x*s
    R[:, 2, 0] = z*x*C - y*s
    R[:, 2, 1] = z*y*C + x*s
    R[:, 2, 2] = c + z*z*C
    R[small]   = _np.eye(3)
    return R


def _perturb_eulers(eu: _np.ndarray, sigma_deg: float) -> _np.ndarray:
    """Apply an independent isotropic small rotation to each ZYZ Euler triple.

    Parameters
    ----------
    eu : numpy.ndarray
        ZYZ Euler angles in radians, shape ``(..., 3)``.
    sigma_deg : float
        Standard deviation of the perturbation rotation magnitude, in degrees.

    Returns
    -------
    numpy.ndarray
        Perturbed Euler angles, same shape and dtype as *eu*.
    """
    shp   = eu.shape
    flat  = _np.ascontiguousarray(eu.reshape(-1, 3), dtype=_np.float64)
    n     = flat.shape[0]
    sigma = _np.deg2rad(sigma_deg)

    axis  = _np.random.randn(n, 3)
    axis /= _np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
    angle = _np.random.randn(n) * sigma
    rvec  = axis * angle[:, None]

    dR  = _rotvec_to_R(rvec)
    R   = _euZYZ_to_R(flat)
    Rp  = _np.einsum('nij,njk->nik', dR, R)
    out = _R_to_euZYZ(Rp).reshape(shp)
    return out.astype(eu.dtype)


def _sigma_schedule(sigma, n: int) -> _np.ndarray:
    """Build a per-entry sequence of perturbation sigmas (degrees).

    Accepts either a fixed scalar (constant schedule) or a ``(lo, hi)`` pair.
    For a range the values are spread evenly over ``[lo, hi]`` with
    :func:`numpy.linspace` and then shuffled, so the whole range is covered
    with low variance while the sigma of a given entry is uncorrelated with its
    position in the buffer (the trainer iterates entries in order).

    Parameters
    ----------
    sigma : float or (float, float)
        Fixed sigma, or ``(lo, hi)`` range in degrees.
    n : int
        Number of entries to generate a schedule for.

    Returns
    -------
    numpy.ndarray
        Length-``n`` array of sigmas in degrees.
    """
    arr = _np.atleast_1d(_np.asarray(sigma, dtype=float))
    if arr.size == 1:
        lo = hi = float(arr[0])
    elif arr.size == 2:
        lo, hi = float(arr[0]), float(arr[1])
    else:
        raise ValueError("sigma must be a scalar or a (lo, hi) pair")
    if lo < 0.0 or hi < lo:
        raise ValueError("sigma range must satisfy 0 <= lo <= hi")

    if lo == hi:
        return _np.full(n, lo)
    if n == 1:
        return _np.array([0.5 * (lo + hi)])
    sched = _np.linspace(lo, hi, n)
    _np.random.shuffle(sched)
    return sched


class VolumePairs():
    """Buffer of independent half-map pairs for Noise2Noise training.

    .. warning:: **Experimental.**  This class is part of the machine-learning
       subpackage and may change or be removed in a future release.

    Half-maps are produced by the SUSAN Averager with random half-set
    assignments, so each pair (half1, half2) shares the same underlying
    signal but has statistically independent noise realisations — the
    ideal N2N training condition.

    The Averager already returns zero-mean, unit-std maps, so no further
    normalisation is applied here.

    Attributes
    ----------
    num_vol : int
        Number of pairs currently stored in the buffer.
    max_vol : int
        Maximum number of pairs the buffer can hold.
    box_size : int
        Full edge length of the input volume in voxels.
    buffer : torch.Tensor or None
        Internal storage tensor of shape ``(max_vol, 2, Z, Y, X)``.
        ``None`` until :meth:`set_size` or :meth:`set_size_mask` is called.
    tmp_base : str
        Path prefix used for temporary files written by :meth:`populate`.
        Default: ``'vol_pair_tmp'``.
    mask : numpy.ndarray or None
        Cropped mask retained by :meth:`set_size_mask`, floored at
        :attr:`min_mask_value`.  ``None`` until :meth:`set_size_mask` is
        called.
    apply_mask : bool
        When ``True``, :meth:`crop_vol` multiplies every cropped volume by
        :attr:`mask` before it is stored in the buffer.  Default: ``False``.
    min_mask_value : float
        Lower bound applied to the mask when it is stored
        (``mask = maximum(min_mask_value, input_mask)``).  A non-zero value
        attenuates — rather than zeroes — voxels outside the mask.
        Default: ``0.0``.
    """

    def __init__(self, tmp_base: str = 'vol_pair_tmp'):
        self.num_vol  = 0
        self.max_vol  = 0
        self.box_size = 0

        self.pz0 = self.pz1 = 0
        self.py0 = self.py1 = 0
        self.px0 = self.px1 = 0

        self.buffer   = None
        self.tmp_base = tmp_base
        self._head    = 0

        self.mask           = None
        self.apply_mask     = False
        self.min_mask_value = 0.0

    # ------------------------------------------------------------------ setup

    def _allocate_buffer(self, num_vol: int):
        shape = (num_vol, 2,
                 self.pz1 - self.pz0,
                 self.py1 - self.py0,
                 self.px1 - self.px0)
        self.buffer  = _torch.zeros(shape, dtype=_torch.float32)
        self.max_vol = num_vol
        self.num_vol = 0
        self._head   = 0

    def set_size(self, vol_size: int, num_vol: int, padding: int):
        """Initialise the buffer with a symmetric fixed-padding crop region.

        Removes *padding* voxels from each face of the cubic volume.  Use
        this when a simple isotropic margin is sufficient.

        Parameters
        ----------
        vol_size : int
            Full edge length of the input volume in voxels.
        num_vol : int
            Number of half-map pairs to allocate space for.
        padding : int
            Number of voxels to crop from each face.
        """
        self.box_size = vol_size
        self.px0 = self.py0 = self.pz0 = padding
        self.px1 = self.py1 = self.pz1 = vol_size - padding
        self.mask       = None
        self.apply_mask = False
        self._allocate_buffer(num_vol)

    def set_size_mask(self, vol_size: int, num_vol: int,
                      mask: _np.ndarray, extra_pad: int = 10,
                      apply_mask: bool = False, min_mask_value: float = 0.0):
        """Initialise the buffer using the tight bounding box of a mask.

        Computes the axis-aligned bounding box of the non-zero voxels in
        *mask*, expands it by *extra_pad* voxels on each side, and uses the
        result as the crop region.  More memory-efficient than
        :meth:`set_size` when the molecule occupies a small fraction of the
        box.

        Parameters
        ----------
        vol_size : int
            Full edge length of the input volume in voxels.
        num_vol : int
            Number of half-map pairs to allocate space for.
        mask : numpy.ndarray
            3-D binary (or non-negative) mask volume.  Non-zero voxels
            define the region of interest.
        extra_pad : int, optional
            Extra voxels added around the bounding box on each side.
            Default: ``10``.
        apply_mask : bool, optional
            If ``True``, :meth:`crop_vol` multiplies every volume by the
            (cropped, floored) mask before it is stored in the buffer.
            Sets :attr:`apply_mask`.  Default: ``False``.
        min_mask_value : float, optional
            Lower bound for the retained mask:
            ``mask = maximum(min_mask_value, mask)``.  With ``0.0`` voxels
            outside the mask are zeroed; a small positive value (e.g. ``0.1``)
            attenuates them instead.  Sets :attr:`min_mask_value`.
            Default: ``0.0``.

        Raises
        ------
        ValueError
            If *mask* contains no non-zero voxels.
        """
        if not _np.any(mask > 0):
            raise ValueError("Mask is empty; cannot determine bounding box")
        pts = _np.where(mask > 0)
        self.box_size = vol_size
        self.pz0 = max(0,         int(pts[0].min()) - extra_pad)
        self.py0 = max(0,         int(pts[1].min()) - extra_pad)
        self.px0 = max(0,         int(pts[2].min()) - extra_pad)
        self.pz1 = min(vol_size,  int(pts[0].max()) + extra_pad + 1)
        self.py1 = min(vol_size,  int(pts[1].max()) + extra_pad + 1)
        self.px1 = min(vol_size,  int(pts[2].max()) + extra_pad + 1)
        self._allocate_buffer(num_vol)

        # Retain the mask so it can be multiplied into each volume on crop.
        self.apply_mask     = bool(apply_mask)
        self.min_mask_value = float(min_mask_value)
        floored             = _np.maximum(self.min_mask_value,
                                          mask).astype(_np.float32)
        self.mask           = _np.ascontiguousarray(self._crop_region(floored))

    # ---------------------------------------------------------------- helpers

    def _crop_region(self, vol: _np.ndarray) -> _np.ndarray:
        """Slice a volume to the active crop region (no masking)."""
        return vol[self.pz0:self.pz1, self.py0:self.py1, self.px0:self.px1]

    def crop_vol(self, vol: _np.ndarray) -> _np.ndarray:
        """Crop a volume to the active region set by :meth:`set_size` or :meth:`set_size_mask`.

        When :attr:`apply_mask` is ``True`` and a mask has been stored, the
        cropped volume is additionally multiplied by :attr:`mask` (floored at
        :attr:`min_mask_value`) before being returned — so volumes are masked
        as :meth:`push` writes them into the buffer.

        Parameters
        ----------
        vol : numpy.ndarray
            Input volume of shape ``(box_size, box_size, box_size)``.

        Returns
        -------
        numpy.ndarray
            Cropped sub-volume, mask-multiplied when :attr:`apply_mask` is set.
        """
        out = self._crop_region(vol)
        if self.apply_mask and self.mask is not None:
            out = out * self.mask
        return out

    def pad_vol(self, vol: _np.ndarray) -> _np.ndarray:
        """Zero-pad a cropped volume back to the full box size.

        Inverse of :meth:`crop_vol`.

        Parameters
        ----------
        vol : numpy.ndarray
            Cropped volume as returned by :meth:`crop_vol`.

        Returns
        -------
        numpy.ndarray
            Full-size volume with the cropped region placed at the correct
            position and zeros elsewhere.
        """
        return _np.pad(vol, ((self.pz0, self.box_size - self.pz1),
                             (self.py0, self.box_size - self.py1),
                             (self.px0, self.box_size - self.px1)))

    # -------------------------------------------------------------- interface

    def __len__(self):
        return self.num_vol

    def __getitem__(self, ix: int):
        if ix >= self.num_vol:
            raise IndexError(ix)
        return self.buffer[ix, 0], self.buffer[ix, 1]

    def __iter__(self):
        for ix in range(self.num_vol):
            yield self[ix]

    def reset(self):
        """Clear the buffer without reallocating memory.

        Sets :attr:`num_vol` to ``0``, resets the write head to slot 0,
        and zeroes the internal tensor.  Subsequent calls to :meth:`push`
        or :meth:`populate` will refill from the beginning.
        """
        self.num_vol = 0
        self._head   = 0
        if self.buffer is not None:
            self.buffer.zero_()

    def push(self, vol1: _np.ndarray, vol2: _np.ndarray):
        """Write one half-map pair into the next circular buffer slot.

        Overwrites the oldest entry once the buffer is full, so
        :attr:`num_vol` never exceeds :attr:`max_vol`.  Use this to
        refresh the training set incrementally across MACE iterations
        without reallocating memory.

        Parameters
        ----------
        vol1 : numpy.ndarray
            First half-map (zero-mean, unit-std), shape
            ``(box_size, box_size, box_size)``.
        vol2 : numpy.ndarray
            Second half-map, same shape.

        Raises
        ------
        RuntimeError
            If the buffer has not been initialised; call :meth:`set_size`
            or :meth:`set_size_mask` first.
        """
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized; call set_size first")
        self.buffer[self._head, 0] = _torch.from_numpy(self.crop_vol(vol1))
        self.buffer[self._head, 1] = _torch.from_numpy(self.crop_vol(vol2))
        self._head   = (self._head + 1) % self.max_vol
        self.num_vol = min(self.num_vol + 1, self.max_vol)

    # ------------------------------------------------------------ population

    def init_with_halfmaps(self, vol1: _np.ndarray, vol2: _np.ndarray):
        """Store a single externally provided half-map pair.

        Resets the buffer and stores the cropped pair at index 0.  Use this
        when half-maps have been reconstructed outside SUSAN and only a single
        pair is needed (e.g. for inference or a quick training run).

        Parameters
        ----------
        vol1 : numpy.ndarray
            First half-map (zero-mean, unit-std).
        vol2 : numpy.ndarray
            Second half-map (zero-mean, unit-std).

        Raises
        ------
        RuntimeError
            If the buffer has not been initialised; call :meth:`set_size`
            or :meth:`set_size_mask` first.
        """
        self.reset()
        self.push(vol1, vol2)

    def save(self, path: str):
        """Save the buffer and all crop/padding metadata to a single file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``'data.vpairs'``).
        """
        _torch.save({
            'buffer':   self.buffer,
            'num_vol':  self.num_vol,
            'max_vol':  self.max_vol,
            'box_size': self.box_size,
            'head':     self._head,
            'pz0': self.pz0, 'pz1': self.pz1,
            'py0': self.py0, 'py1': self.py1,
            'px0': self.px0, 'px1': self.px1,
            'mask':           (None if self.mask is None
                               else _torch.from_numpy(self.mask)),
            'apply_mask':     self.apply_mask,
            'min_mask_value': self.min_mask_value,
        }, path)

    @classmethod
    def load(cls, path: str, tmp_base: str = 'vol_pair_tmp') -> 'VolumePairs':
        """Reconstruct a VolumePairs from a file saved with :meth:`save`.

        Parameters
        ----------
        path : str
            Path to a file previously written by :meth:`save`.
        tmp_base : str, optional
            ``tmp_base`` for the reconstructed object.  Default:
            ``'vol_pair_tmp'``.

        Returns
        -------
        VolumePairs
        """
        ck  = _torch.load(path, map_location='cpu', weights_only=True)
        obj = cls(tmp_base=tmp_base)
        obj.buffer   = ck['buffer']
        obj.num_vol  = int(ck['num_vol'])
        obj.max_vol  = int(ck['max_vol'])
        obj.box_size = int(ck['box_size'])
        obj._head    = int(ck['head'])
        obj.pz0 = int(ck['pz0']); obj.pz1 = int(ck['pz1'])
        obj.py0 = int(ck['py0']); obj.py1 = int(ck['py1'])
        obj.px0 = int(ck['px0']); obj.px1 = int(ck['px1'])
        m = ck.get('mask', None)
        obj.mask           = None if m is None else m.numpy()
        obj.apply_mask     = bool(ck.get('apply_mask', False))
        obj.min_mask_value = float(ck.get('min_mask_value', 0.0))
        return obj

    def populate(self, avgr, ptcls, tomo_filename: str,
                 num_entries: int = None,
                 sigma_ang_3D=0.0, sigma_ang_2D=0.0, sigma_def=0.0):
        """Fill the buffer by re-randomising half-set assignments and running the Averager.

        Each iteration re-randomises the ``half_id`` array in *ptcls*, saves
        a temporary particles file, runs the Averager to produce a pair of
        half-maps, and stores the cropped pair in the buffer.  The loop
        repeats until the buffer is full or *num_entries* new pairs have been
        added.

        When *sigma_ang_3D* or *sigma_ang_2D* is non-zero, the orientations of
        the **half-1** particles are perturbed by an independent isotropic small
        rotation before reconstruction, so ``vol1`` (buffer slot 0) becomes a
        deliberately misaligned reconstruction while ``vol2`` (slot 1) stays
        cleanly aligned.  Train with :meth:`Noise2NoiseTrainer.train`
        ``directional=True`` on such pairs to teach the network to recover from
        small alignment errors (slot 0 = input, slot 1 = target).

        The Averager's ``verbosity`` and ``rec_halfsets`` attributes, and the
        original ``half_id``, ``ali_eu`` and ``prj_eu`` arrays, are fully
        restored on exit even if an exception is raised.

        Parameters
        ----------
        avgr : :class:`~susan.modules.Averager`
            Configured Averager instance.  ``rec_halfsets`` will be
            temporarily forced to ``True``; ``verbosity`` will be silenced.
        ptcls : :class:`~susan.data.Particles`
            Particle stack.  The ``half_id`` array (and, when perturbation is
            enabled, ``ali_eu`` / ``prj_eu``) is temporarily modified and
            restored on exit.
        tomo_filename : str
            Path to the ``.tomostxt`` tomograms file.
        num_entries : int, optional
            Maximum number of new pairs to add in this call.  ``None``
            fills the buffer to capacity.  Default: ``None``.
        sigma_ang_3D : float or (float, float), optional
            Std. dev. (in degrees) of the isotropic rotation applied to the
            half-1 particles' 3-D orientations (``ali_eu``).  Models 3-D
            alignment error.  A scalar uses a fixed sigma for every entry; a
            ``(lo, hi)`` pair sweeps the range across the entries generated in
            this call (evenly spread and shuffled), which trains the network to
            be robust over a range of blur levels rather than a single one.
            Include ``lo == 0`` to also see cleanly-aligned pairs and avoid
            over-sharpening at inference.  ``0.0`` disables it.  Default: ``0.0``.
        sigma_ang_2D : float or (float, float), optional
            Same as *sigma_ang_3D*, but applied independently to each half-1
            projection orientation (``prj_eu``).  Models per-tilt / tilt-series
            alignment error.  ``0.0`` disables it.  Default: ``0.0``.
        sigma_def : float or (float, float), optional
            Std. dev. (in Angstroms) of an additive Gaussian shift applied
            independently to each half-1 projection's defocus.  The same shift
            is added to ``def_U`` and ``def_V`` so the mean defocus moves while
            the astigmatism is preserved.  Models defocus-estimation error,
            which degrades CTF correction.  Scalar or ``(lo, hi)`` range, same
            semantics as *sigma_ang_3D*.  ``0.0`` disables it.  Default: ``0.0``.

        Raises
        ------
        RuntimeError
            If the buffer has not been initialised; call :meth:`set_size`
            or :meth:`set_size_mask` first.

        Examples
        --------
        Plain Noise2Noise denoising pairs (both halves cleanly aligned):

        >>> vp = VolumePairs()
        >>> vp.set_size_mask(box_size, num_vol=50, mask=mask)
        >>> vp.populate(avgr, ptcls, 'tomos.tomostxt')
        >>> trainer.train(vp, n_epochs=100)          # symmetric, denoising only

        Misalignment-recovery pairs: perturb the half-1 orientations so slot 0
        is a misaligned reconstruction and slot 1 is cleanly aligned, then train
        directionally (slot 0 -> slot 1) so the network learns to undo the
        alignment blur while still denoising:

        >>> vp.populate(avgr, ptcls, 'tomos.tomostxt', sigma_ang_3D=3.0)
        >>> trainer.train(vp, n_epochs=100, directional=True)

        For robustness across blur levels (recommended), pass a ``(lo, hi)``
        range including ``0`` instead of a single sigma, so the buffer spans
        cleanly-aligned to strongly-misaligned pairs:

        >>> vp.populate(avgr, ptcls, 'tomos.tomostxt', sigma_ang_3D=(0.0, 4.0))
        >>> trainer.train(vp, n_epochs=100, directional=True)

        ``sigma_ang_3D`` is the std. dev. of the applied rotation *magnitude* in
        degrees, so the typical geodesic error is roughly ``0.8 * sigma`` (about
        2.3 deg for ``sigma_ang_3D=3.0``).  Add ``sigma_ang_2D`` to also perturb
        each projection orientation independently (per-tilt alignment error), or
        ``sigma_def`` (Angstroms) to perturb the per-projection defocus and model
        CTF-correction error.  Enable and validate these one at a time rather
        than all at once.
        """
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized; call set_size first")

        import susan as _susan

        limit     = self.max_vol if num_entries is None else num_entries
        sched_3D  = _sigma_schedule(sigma_ang_3D, limit)
        sched_2D  = _sigma_schedule(sigma_ang_2D, limit)
        sched_def = _sigma_schedule(sigma_def,    limit)
        perturb   = bool(_np.any(sched_3D  > 0.0) or
                         _np.any(sched_2D  > 0.0) or
                         _np.any(sched_def > 0.0))

        half_id      = _np.copy(ptcls.half_id)
        ali_eu0      = _np.copy(ptcls.ali_eu) if perturb else None
        prj_eu0      = _np.copy(ptcls.prj_eu) if perturb else None
        def_U0       = _np.copy(ptcls.def_U)  if perturb else None
        def_V0       = _np.copy(ptcls.def_V)  if perturb else None
        verbosity    = avgr.verbosity
        rec_halfmaps = avgr.rec_halfsets
        avgr.verbosity    = 0
        avgr.rec_halfsets = True

        try:
            for count in range(limit):
                if perturb:
                    # Reset to the pristine values so perturbations do not
                    # accumulate across iterations.
                    ptcls.ali_eu[:] = ali_eu0
                    ptcls.prj_eu[:] = prj_eu0
                    ptcls.def_U[:]  = def_U0
                    ptcls.def_V[:]  = def_V0
                ptcls.halfsets_randomize()
                if perturb:
                    sel = ptcls.half_id == 1
                    if sched_3D[count] > 0.0:
                        for r in range(ptcls.ali_eu.shape[0]):
                            ptcls.ali_eu[r, sel, :] = _perturb_eulers(
                                ptcls.ali_eu[r, sel, :], sched_3D[count])
                    if sched_2D[count] > 0.0:
                        ptcls.prj_eu[sel, :, :] = _perturb_eulers(
                            ptcls.prj_eu[sel, :, :], sched_2D[count])
                    if sched_def[count] > 0.0:
                        # Same shift on U and V: moves mean defocus, keeps
                        # astigmatism.  Independent per (particle, projection).
                        d = (_np.random.randn(*ptcls.def_U[sel, :].shape)
                             * sched_def[count]).astype(ptcls.def_U.dtype)
                        ptcls.def_U[sel, :] += d
                        ptcls.def_V[sel, :] += d
                ptcls.save(f'{self.tmp_base}.ptclsraw')
                avgr.reconstruct(self.tmp_base, tomo_filename,
                                 f'{self.tmp_base}.ptclsraw', self.box_size)
                vol1 = -_susan.read(f'{self.tmp_base}_class001_half1.mrc')
                vol2 = -_susan.read(f'{self.tmp_base}_class001_half2.mrc')
                self.push(vol1, vol2)
        finally:
            ptcls.half_id[:]  = half_id
            if perturb:
                ptcls.ali_eu[:] = ali_eu0
                ptcls.prj_eu[:] = prj_eu0
                ptcls.def_U[:]  = def_U0
                ptcls.def_V[:]  = def_V0
            avgr.verbosity    = verbosity
            avgr.rec_halfsets = rec_halfmaps
