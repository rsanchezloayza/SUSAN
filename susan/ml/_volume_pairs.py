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
                 num_entries: int = None):
        """Fill the buffer by re-randomising half-set assignments and running the Averager.

        Each iteration re-randomises the ``half_id`` array in *ptcls*, saves
        a temporary particles file, runs the Averager to produce a pair of
        half-maps, and stores the cropped pair in the buffer.  The loop
        repeats until the buffer is full or *num_entries* new pairs have been
        added.

        The Averager's ``verbosity`` and ``rec_halfsets`` attributes, and the
        original ``half_id`` array, are fully restored on exit even if an
        exception is raised.

        Parameters
        ----------
        avgr : :class:`~susan.modules.Averager`
            Configured Averager instance.  ``rec_halfsets`` will be
            temporarily forced to ``True``; ``verbosity`` will be silenced.
        ptcls : :class:`~susan.data.Particles`
            Particle stack.  The ``half_id`` array is temporarily modified
            and restored on exit.
        tomo_filename : str
            Path to the ``.tomostxt`` tomograms file.
        num_entries : int, optional
            Maximum number of new pairs to add in this call.  ``None``
            fills the buffer to capacity.  Default: ``None``.

        Raises
        ------
        RuntimeError
            If the buffer has not been initialised; call :meth:`set_size`
            or :meth:`set_size_mask` first.
        """
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized; call set_size first")

        import susan as _susan

        half_id      = _np.copy(ptcls.half_id)
        verbosity    = avgr.verbosity
        rec_halfmaps = avgr.rec_halfsets
        avgr.verbosity    = 0
        avgr.rec_halfsets = True

        try:
            count = 0
            limit = self.max_vol if num_entries is None else num_entries
            while count < limit:
                ptcls.halfsets_randomize()
                ptcls.save(f'{self.tmp_base}.ptclsraw')
                avgr.reconstruct(self.tmp_base, tomo_filename,
                                 f'{self.tmp_base}.ptclsraw', self.box_size)
                vol1 = -_susan.read(f'{self.tmp_base}_class001_half1.mrc')
                vol2 = -_susan.read(f'{self.tmp_base}_class001_half2.mrc')
                self.push(vol1, vol2)
                count += 1
        finally:
            ptcls.half_id[:]  = half_id
            avgr.verbosity    = verbosity
            avgr.rec_halfsets = rec_halfmaps
