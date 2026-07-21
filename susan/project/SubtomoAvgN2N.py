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

"""Subtomogram averaging with per-iteration Noise2Noise denoising.

.. warning::

   **Experimental / in development.**  The API, default training
   hyper-parameters, and per-iteration behaviour are not yet stable.
   Results have not been validated across datasets and the underlying
   :mod:`susan.ml` denoiser is itself experimental.  Expect breaking
   changes between releases.  Do not rely on this class for production
   workflows.

:class:`SubtomoAvgN2N` extends :class:`~susan.project.SubtomoAvgSched.SubtomoAvgSched`
with Noise2Noise denoising applied after every reconstruction.  The model
and trainer are recreated from stored parameters each iteration so that GPU
memory is freed between the training step and the next alignment.

Typical manual loop::

    import susan

    sta = susan.project.SubtomoAvgN2N('prj_001', 128)
    sta.tomogram_file     = 'tomos.tomostxt'
    sta.initial_reference = 'ref.refstxt'
    sta.initial_particles = 'ptcls.ptclsraw'
    sta.list_gpus_ids     = [0, 1]
    sta.n2n_mask          = 'mask.mrc'

    tomos = susan.data.Tomograms(sta.tomogram_file)

    # Pre-train on initial particles
    ptcls = susan.data.Particles(sta.initial_particles)
    susan.data.Particles.Geom.enable_by_tilt(ptcls, tomos, tilt_deg_max=19)
    sta.create_training_dataset(ptcls, fine_tune=False)
    vol = sta.train_and_denoise(0, fine_tune=False)
    susan.io.mrc.write(vol, 'initial_denoised.mrc', sta.pix_size)

    # Refinement loop
    for i in range(1, 11):
        bp = sta.run_iteration(i)
        ptcls = sta.get_ptcls(i)
        susan.data.Particles.Geom.enable_by_tilt(ptcls, tomos, tilt_deg_max=19)
        sta.create_training_dataset(ptcls, fine_tune=True)
        susan.io.mrc.write(sta.train_and_denoise(i), sta.path_map(i), sta.pix_size)

Automated loop via schedulers::

    sta.bandpass_scheduler       = susan.project.Schedulers.Bandpass.Adaptive(128, max_step=2.5)
    sta.iteration_type_scheduler = susan.project.Schedulers.IterationType.Fixed('2D')
    sta.run(1, 10)
"""

import gc        as _gc
import numpy     as _np

import susan.data    as _ssa_data
# NOTE: susan.ml (and its torch dependency) is imported lazily inside the
# methods that need it, so that importing susan without PyTorch installed
# still works and simply disables the ML features.

from os      import mkdir  as _mkdir
from os.path import exists as _file_exists

from susan.io.mrc import read  as _mrc_read
from susan.io.mrc import write as _mrc_write

from susan.project.SubtomoAvgSched import SubtomoAvgSched as _SubtomoAvgSched


class SubtomoAvgN2N(_SubtomoAvgSched):
    """Subtomogram averaging with Noise2Noise denoising each iteration.

    .. warning::

       **Experimental / in development.**  API and default hyper-parameters
       may change; results have not been broadly validated.  Not recommended
       for production workflows.

    Extends :class:`~susan.project.SubtomoAvgSched.SubtomoAvgSched` by
    overriding :meth:`run_postprocessing` to train a
    :class:`~susan.ml.Noise2Noise` model on the freshly reconstructed
    half-maps and replace the reference map with the denoised output.

    The model and trainer are **not** stored as persistent instance
    attributes.  Their construction parameters are stored, and the objects
    are re-instantiated each call so that GPU memory is released after
    training.

    Parameters
    ----------
    prj_name : str
        Project directory.
    box_size : int, optional
        Box size in pixels.  Reads ``info.prjtxt`` when omitted.
    n2n_arch : str, optional
        Backbone architecture: ``'convnet'`` or ``'unet'``.  Default:
        ``'convnet'``.
    n2n_depth : int, optional
        For ``'convnet'``: total number of blocks. For ``'unet'``: number
        of encoder/decoder levels.  Default: ``6``.
    n2n_n_feat : int, optional
        Number of feature maps per layer.  Default: ``32``.
    n2n_loss : str, optional
        ``'l1'`` (MAE) or ``'l2'`` (MSE).  Default: ``'l1'``.

    Attributes
    ----------
    n2n_arch, n2n_depth, n2n_n_feat, n2n_loss
        Architecture / loss parameters saved for model recreation.
    n2n_scratch_lr, n2n_scratch_epochs, n2n_scratch_n_pairs : float / int
        Training parameters used when no previous model exists.
    n2n_finetune_lr, n2n_finetune_epochs, n2n_finetune_n_pairs : float / int
        Training parameters used for fine-tuning from a previous model.
    n2n_batch_size : int
        Pairs per gradient step.  Default: ``1``.
    n2n_print_every : int
        Print loss every N epochs (``0`` to silence).  Default: ``10``.
    n2n_mask : str or None
        Path to an MRC file used as mask for
        :meth:`~susan.ml.VolumePairs.set_size_mask`.  The file is read
        each time a new :class:`~susan.ml.VolumePairs` buffer is
        allocated.  If ``None``, :meth:`~susan.ml.VolumePairs.set_size`
        is used instead with ``padding=0``.
    n2n_extra_pad : int
        Extra padding for :meth:`~susan.ml.VolumePairs.set_size_mask`.
        Default: ``0``.
    n2n_num_entries : int or None
        Passed to :meth:`~susan.ml.VolumePairs.populate` as ``num_entries``.
        ``None`` fills to capacity.
    n2n_free_gpu : bool
        If ``True`` (default), delete model and trainer after each call to
        :meth:`train_and_denoise` and empty the CUDA cache.
    n2n_ptcls_modifier : callable or None
        Optional ``(ptcls, tomos) -> None`` hook applied to particles in
        :meth:`run_postprocessing` (automated loop only) **after** the
        standard tilt-limit logic.  ``None`` means no extra modification.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, prj_name, box_size=None,
                 n2n_arch='convnet', n2n_depth=6, n2n_n_feat=32,
                 n2n_loss='l1'):
        super().__init__(prj_name, box_size)

        # Architecture
        self.n2n_arch   = n2n_arch
        self.n2n_depth  = n2n_depth
        self.n2n_n_feat = n2n_n_feat
        self.n2n_loss   = n2n_loss

        # Scratch training (no previous model)
        self.n2n_scratch_lr      = 1e-4
        self.n2n_scratch_epochs  = 60
        self.n2n_scratch_n_pairs = 40

        # Fine-tuning
        self.n2n_finetune_lr      = 1e-5
        self.n2n_finetune_epochs  = 10
        self.n2n_finetune_n_pairs = 10

        # Optimization
        self.n2n_batch_size  = 1
        self.n2n_print_every = 10

        # VolumePairs geometry
        self.n2n_mask        = None
        self.n2n_extra_pad   = 0
        self.n2n_num_entries = None

        # Memory and particle hooks
        self.n2n_free_gpu        = True
        self.n2n_ptcls_modifier  = None

        # Always preserve the raw reconstruction when N2N is active
        self.save_raw_map = True

        # Create the scratch directory for VolumePairs temp files
        tmp_dir = self.prj_name + '/n2n_tmp'
        if not _file_exists(tmp_dir):
            _mkdir(tmp_dir)

    # ------------------------------------------------------------------
    # Internal factories
    # ------------------------------------------------------------------

    def _tmp_base(self):
        return self.prj_name + '/n2n_tmp/vol_pair_tmp'

    def _initial_model_path(self):
        return self.prj_name + '/initial_model.pth'

    def _make_model(self, weights_path=None):
        """Create a Noise2Noise model, optionally loading weights."""
        import susan.ml as _ssa_ml
        if weights_path is not None and _file_exists(weights_path):
            return _ssa_ml.Noise2Noise.load(weights_path,
                                            gpus=self.list_gpus_ids)
        m = _ssa_ml.Noise2Noise(
            arch=self.n2n_arch,
            depth=self.n2n_depth,
            n_feat=self.n2n_n_feat,
            gpus=self.list_gpus_ids,
        )
        m.reset_weights()
        return m

    def _make_trainer(self, model):
        """Create a Noise2NoiseTrainer with the configured loss."""
        import susan.ml as _ssa_ml
        return _ssa_ml.Noise2NoiseTrainer(model, loss=self.n2n_loss)

    def _free_gpu(self, model, trainer):
        """Delete objects and release GPU memory."""
        import torch as _torch
        del model, trainer
        _gc.collect()
        for k in range(_torch.cuda.device_count()):
            with _torch.cuda.device(k):
                _torch.cuda.empty_cache()

    def _resolve_model_path(self, prv):
        """Return the best available weights path, or None for fresh weights.

        Priority: previous iteration model → initial model → None.
        """
        if prv.ite_dir and _file_exists(prv.ite_dir + 'model.pth'):
            return prv.ite_dir + 'model.pth'
        ip = self._initial_model_path()
        if _file_exists(ip):
            return ip
        return None

    def _make_volume_pairs(self, n_pairs):
        """Allocate a VolumePairs buffer with the project's tmp_base."""
        import susan.ml as _ssa_ml
        data = _ssa_ml.VolumePairs(tmp_base=self._tmp_base())
        if self.n2n_mask is not None:
            mask, _ = _mrc_read(self.n2n_mask)
            data.set_size_mask(self.box_size, n_pairs,
                               mask=mask,
                               extra_pad=self.n2n_extra_pad)
        else:
            data.set_size(self.box_size, n_pairs, padding=0)
        return data

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_training_dataset(self, ptcls, fine_tune=True):
        """Build (or rebuild) the internal VolumePairs training buffer.

        The caller is responsible for any particle modifications (tilt
        limiting, flag zeroing, etc.) before passing *ptcls*.

        Parameters
        ----------
        ptcls : :class:`~susan.data.Particles`
            Particle stack to reconstruct from.  Modified temporarily
            during population but restored on exit.
        fine_tune : bool, optional
            If ``True`` (default) use ``n2n_finetune_n_pairs``.
            If ``False`` use ``n2n_scratch_n_pairs``.
        """
        n_pairs = (self.n2n_finetune_n_pairs if fine_tune
                   else self.n2n_scratch_n_pairs)
        self._data = self._make_volume_pairs(n_pairs)
        self._data.populate(self.averager, ptcls, self.tomogram_file,
                            num_entries=self.n2n_num_entries)

    def _train_and_denoise_files(self, cur, prv, fine_tune=True):
        """Core train+denoise using _IterationFiles objects.

        Used internally by both :meth:`train_and_denoise` and
        :meth:`run_postprocessing`.
        """
        lr     = self.n2n_finetune_lr     if fine_tune else self.n2n_scratch_lr
        epochs = self.n2n_finetune_epochs if fine_tune else self.n2n_scratch_epochs

        model   = self._make_model(self._resolve_model_path(prv))
        trainer = self._make_trainer(model)

        trainer.train(self._data,
                      n_epochs=epochs,
                      lr=lr,
                      batch_size=self.n2n_batch_size,
                      print_every=self.n2n_print_every)

        model.save(cur.ite_dir + 'model.pth')

        tmp_map  = self._tmp_base() + '_class001.mrc'
        vol, _   = _mrc_read(tmp_map)
        denoised = self._data.pad_vol(
            trainer.predict(self._data.crop_vol(-vol))
        )
        result = -denoised.astype(_np.float32)

        if self.n2n_free_gpu:
            self._free_gpu(model, trainer)

        return result

    def train_and_denoise(self, ite, fine_tune=True):
        """Train the N2N model on the current dataset and denoise the map.

        Loads weights from the previous iteration (or initial model if
        available, else fresh weights), trains, saves the model to
        ``<ite_dir>/model.pth``, denoises the map, and returns the
        inverted denoised volume ready to write to disk.

        :meth:`create_training_dataset` must be called before this method.

        Parameters
        ----------
        ite : int
            Current iteration number.
        fine_tune : bool, optional
            If ``True`` (default) use finetune lr/epochs.
            If ``False`` use scratch lr/epochs.

        Returns
        -------
        numpy.ndarray
            Denoised map, sign-inverted and padded to full box size.
        """
        cur = self.iteration_files(ite)
        prv = self.iteration_files(ite - 1)
        return self._train_and_denoise_files(cur, prv, fine_tune)

    # ------------------------------------------------------------------
    # Automated loop — override run_postprocessing
    # ------------------------------------------------------------------

    def run_postprocessing(self, cur, prv):
        """FSC reporting + N2N denoising for the automated :meth:`run` loop.

        Calls the parent postprocessing (FSC, resolution reporting, raw map
        save), then builds the VolumePairs dataset from the current iteration
        particles (applying tilt limits and :attr:`n2n_ptcls_modifier`),
        trains the model, and overwrites the reference map with the denoised
        output.

        Returns
        -------
        float
            Resolution estimate in Fourier pixels.
        """
        bp_est = super().run_postprocessing(cur, prv)

        # --- Particle preparation (mirrors select_particles tilt logic) ---
        ptcls = _ssa_data.Particles(cur.ptcl_rslt)

        v = self.max_tilt_reconstruction
        tilt_active = (v is not None) and (
            isinstance(v, (list, tuple, _np.ndarray)) or v >= 0)

        tomos = None
        if tilt_active:
            tomos = _ssa_data.Tomograms(self.tomogram_file)
            if isinstance(v, (list, tuple, _np.ndarray)):
                v = _np.asarray(v, dtype=_np.float32)
                _ssa_data.Particles.Geom.enable_by_tilt_range(
                    ptcls, tomos,
                    tilt_deg_min=float(v.min()),
                    tilt_deg_max=float(v.max()))
            else:
                _ssa_data.Particles.Geom.enable_by_tilt(
                    ptcls, tomos, tilt_deg_max=float(v))

        if self.n2n_ptcls_modifier is not None:
            if tomos is None:
                tomos = _ssa_data.Tomograms(self.tomogram_file)
            self.n2n_ptcls_modifier(ptcls, tomos)

        # Build dataset and train+denoise using cur/prv directly
        self.create_training_dataset(ptcls, fine_tune=True)
        denoised = self._train_and_denoise_files(cur, prv, fine_tune=True)

        map_file = cur.ite_dir + 'map_class001.mrc'
        _mrc_write(denoised, map_file, self.pix_size)

        return bp_est
