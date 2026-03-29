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

    # ------------------------------------------------------------------ setup

    def _allocate_buffer(self, num_vol: int):
        shape = (num_vol, 2,
                 self.pz1 - self.pz0,
                 self.py1 - self.py0,
                 self.px1 - self.px0)
        self.buffer  = _torch.zeros(shape, dtype=_torch.float32)
        self.max_vol = num_vol
        self.num_vol = 0

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
        self._allocate_buffer(num_vol)

    def set_size_mask(self, vol_size: int, num_vol: int,
                      mask: _np.ndarray, extra_pad: int = 10):
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

    # ---------------------------------------------------------------- helpers

    def crop_vol(self, vol: _np.ndarray) -> _np.ndarray:
        """Crop a volume to the active region set by :meth:`set_size` or :meth:`set_size_mask`.

        Parameters
        ----------
        vol : numpy.ndarray
            Input volume of shape ``(box_size, box_size, box_size)``.

        Returns
        -------
        numpy.ndarray
            Cropped sub-volume.
        """
        return vol[self.pz0:self.pz1, self.py0:self.py1, self.px0:self.px1]

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

        Sets :attr:`num_vol` to ``0`` and zeroes the internal tensor.
        Subsequent calls to :meth:`populate` will refill from index 0.
        """
        self.num_vol = 0
        if self.buffer is not None:
            self.buffer.zero_()

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
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized; call set_size first")
        self.reset()
        self.buffer[0, 0] = _torch.from_numpy(self.crop_vol(vol1))
        self.buffer[0, 1] = _torch.from_numpy(self.crop_vol(vol2))
        self.num_vol = 1

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
            for ix in range(self.num_vol, self.max_vol):
                ptcls.halfsets_randomize()
                ptcls.save(f'{self.tmp_base}.ptclsraw')
                avgr.reconstruct(self.tmp_base, tomo_filename,
                                 f'{self.tmp_base}.ptclsraw', self.box_size)
                vol1 = -_susan.read(f'{self.tmp_base}_class001_half1.mrc')
                vol2 = -_susan.read(f'{self.tmp_base}_class001_half2.mrc')
                self.buffer[ix, 0] = _torch.from_numpy(self.crop_vol(vol1))
                self.buffer[ix, 1] = _torch.from_numpy(self.crop_vol(vol2))
                self.num_vol += 1
                count += 1
                if num_entries is not None and count >= num_entries:
                    break
        finally:
            ptcls.half_id[:]  = half_id
            avgr.verbosity    = verbosity
            avgr.rec_halfsets = rec_halfmaps
