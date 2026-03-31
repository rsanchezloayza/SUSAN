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

__all__ = ['Noise2Moments', 'Noise2MomentsTrainer']

import numpy as _np
import torch    as _torch
import torch.nn as _nn
import torch.nn.functional as _F


class Noise2Moments(_nn.Module):
    """Heteroscedastic Noise2Noise denoiser for 3-D cryo-EM volumes.

    .. warning:: **Experimental.**  This class is part of the machine-learning
       subpackage and may change or be removed in a future release.

    Predicts per-voxel mean and log-variance from a noisy half-map,
    trained against the paired half-map as the N2N target.  Because both
    half-maps are independent noisy realisations of the same underlying
    signal, the optimal N2N predictor converges to the clean signal mean
    and the half-map noise variance.

    Architecture
    ------------
    Sequential 3-D fully-convolutional network with 3×3×3 kernels.
    The first block (Conv3d + ReLU) and the final block (Conv3d to 2
    channels) have no normalisation; intermediate blocks use
    ``InstanceNorm3d``, which computes statistics over spatial dimensions
    and is unaffected by batch size, making it correct for single-volume
    batches.

    The two output channels are split into *mean* and *log-variance*.
    The log-variance is clamped to ``[logvar_min, logvar_max]`` to
    prevent numerical overflow in ``exp(logvar)``.

    Multi-GPU support
    -----------------
    Convolutional blocks are distributed evenly across *gpus*.
    Activations are moved between devices in :meth:`forward`.

    Persistence
    -----------
    Use :meth:`save` / :meth:`load` to store and restore a model.  Both
    methods bundle the architecture hyperparameters with the weights in a
    single file, so the model can be reconstructed after a kernel restart
    without needing to remember *depth*, *n_feat*, or *logvar_clamp*.

    Parameters
    ----------
    depth : int
        Total number of convolutional blocks (≥ 2).  The first and last
        blocks are fixed; ``depth − 2`` intermediate blocks with
        ``InstanceNorm3d`` are inserted between them.  Default: ``6``.
    n_feat : int
        Feature channels in the intermediate blocks.  Default: ``32``.
    gpus : list of int
        CUDA device indices.  Blocks are split evenly; at least one GPU
        is required.  Default: ``[0]``.
    logvar_clamp : tuple of float
        ``(min, max)`` clamp range for the log-variance output channel.
        Default: ``(-10.0, 10.0)``.
    """

    def __init__(self, depth: int = 6, n_feat: int = 32,
                 gpus: list = [0],
                 logvar_clamp: tuple = (-10.0, 10.0)):
        super().__init__()

        assert isinstance(depth, int)  and depth  >= 2, "depth must be int >= 2"
        assert isinstance(n_feat, int) and n_feat  > 0, "n_feat must be positive"
        assert isinstance(gpus, (list, tuple)) and len(gpus) >= 1

        self.depth  = depth
        self.n_feat = n_feat

        self.logvar_min, self.logvar_max = logvar_clamp
        assert self.logvar_min < self.logvar_max

        self.gpus    = list(gpus)
        self.devices = [_torch.device(f"cuda:{i}") for i in self.gpus]

        common = {"kernel_size": 3, "padding": 1}
        blocks = []
        blocks.append(_nn.Sequential(
            _nn.Conv3d(1, n_feat, **common),
            _nn.ReLU(inplace=True),
        ))
        for _ in range(depth - 2):
            blocks.append(_nn.Sequential(
                _nn.Conv3d(n_feat, n_feat, **common),
                _nn.InstanceNorm3d(n_feat, affine=True),  # correct with batch=1
                _nn.ReLU(inplace=True),
            ))
        blocks.append(_nn.Conv3d(n_feat, 2, 3, padding=1))

        self.parts = _nn.ModuleList()
        n_blocks   = len(blocks)
        n_parts    = len(self.devices)
        splits     = _torch.linspace(0, n_blocks, n_parts + 1, dtype=_torch.int32)

        for i in range(n_parts):
            part_blocks = blocks[splits[i]:splits[i + 1]]
            assert len(part_blocks) > 0, \
                f"Too many GPUs ({n_parts}) for chosen depth ({depth}); " \
                f"need at most {n_blocks} GPUs"
            self.parts.append(_nn.Sequential(*part_blocks).to(self.devices[i]))

    # ---------------------------------------------------------------- devices

    def input_device(self) -> _torch.device:
        """Return the device of the first block (expected input device)."""
        return self.devices[0]

    def output_device(self) -> _torch.device:
        """Return the device of the last block (output device)."""
        return self.devices[-1]

    # --------------------------------------------------------------- forward

    def forward(self, x: _torch.Tensor):
        """Run a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(N, 1, D, H, W)``.  Must be on
            :meth:`input_device`.

        Returns
        -------
        mean : torch.Tensor
            Predicted signal mean, shape ``(N, 1, D, H, W)``, on
            :meth:`output_device`.
        logvar : torch.Tensor
            Predicted log noise variance, same shape and device, clamped
            to ``[logvar_min, logvar_max]``.
        """
        assert x.ndim == 5 and x.size(1) == 1, \
            f"Expected input (N,1,D,H,W), got {tuple(x.shape)}"

        x = x.to(self.devices[0])
        for i, part in enumerate(self.parts):
            x = part(x)
            if i + 1 < len(self.parts):
                x = x.to(self.devices[i + 1])

        mean, logvar = _torch.split(x, 1, dim=1)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        return mean, logvar

    # -------------------------------------------------------------- utilities

    def reset_weights(self):
        """Re-initialise all learnable parameters to their default values."""
        def _reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(_reset)

    def save(self, path: str):
        """Save architecture hyperparameters and weights to a single file.

        Bundles *depth*, *n_feat*, *logvar_clamp*, and the full
        ``state_dict`` in one file.  Use :meth:`load` to reconstruct
        the model after a kernel restart.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``'model.pth'``).
        """
        _torch.save({
            'depth':      self.depth,
            'n_feat':     self.n_feat,
            'logvar_min': self.logvar_min,
            'logvar_max': self.logvar_max,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, gpus: list = [0]) -> 'Noise2Moments':
        """Reconstruct a model from a file saved with :meth:`save`.

        The *gpus* argument controls placement; the original training GPUs
        are not required.  Weights are copied to whichever devices the
        reconstructed model assigns its blocks to.

        Parameters
        ----------
        path : str
            Path to a file previously written by :meth:`save`.
        gpus : list of int, optional
            CUDA device indices for the reconstructed model.
            Default: ``[0]``.

        Returns
        -------
        Noise2Moments
            Fully initialised model with loaded weights.
        """
        ckpt  = _torch.load(path, map_location='cpu', weights_only=True)
        model = cls(
            depth        = int(ckpt['depth']),
            n_feat       = int(ckpt['n_feat']),
            gpus         = gpus,
            logvar_clamp = (float(ckpt['logvar_min']), float(ckpt['logvar_max'])),
        )
        model.load_state_dict(ckpt['state_dict'])
        return model


# ---------------------------------------------------------------------------

class Noise2MomentsTrainer:
    """Training and inference wrapper for :class:`Noise2Moments`.

    .. warning:: **Experimental.**  This class is part of the machine-learning
       subpackage and may change or be removed in a future release.

    Primary loss
    ------------
    Heteroscedastic Gaussian NLL (default)::

        loss = 0.5 * (exp(−logvar) * (y − mean)² + logvar)

    When :attr:`use_student_t` is ``True``, a Student-t NLL is used
    instead.  Its heavier tails down-weight voxels with large half-map
    disagreement (flexible regions, outliers), producing a more
    conservative predicted mean.  Degrees of freedom are set by
    :attr:`student_t_nu` (lower = heavier tails).

    Optional auxiliary losses
    -------------------------
    Enabled by setting the corresponding key in :attr:`loss_weights` to a
    positive value (all default to ``0.0``):

    .. list-table::
       :header-rows: 1
       :widths: 20 80

       * - Key
         - Effect
       * - ``'tv'``
         - L1 total variation on the predicted mean; promotes sharp edges.
       * - ``'smoothness'``
         - L2 squared-gradient smoothness; promotes spatially smooth output.
       * - ``'sparsity'``
         - L1 penalty on the predicted mean; suppresses all amplitudes.
       * - ``'local_var'``
         - Local spatial variance of the predicted mean, optionally
           uncertainty-weighted by ``exp(logvar)`` so smoothing pressure
           focuses on flexible regions.
       * - ``'var_prior'``
         - Gaussian prior on logvar; prevents runaway uncertainty in
           low-signal regions.
       * - ``'outlier'``
         - Hinge squared penalty on predicted mean values beyond
           :attr:`outlier_threshold`.  Zero inside the normal signal
           range; does not penalise normal-amplitude voxels.
       * - ``'positivity'``
         - One-sided L1 hinge: ``relu(−mean).mean()``.  Softly
           encourages non-negative output without hard constraints.

    Attributes
    ----------
    model : :class:`Noise2Moments`
        The wrapped network.
    hard_logvar_floor : float or None
        If set, clamps the predicted logvar from below before loss
        evaluation and inference.  Default: ``None`` (disabled).
    loss_weights : dict
        Per-term auxiliary loss weights (see table above).  All default
        to ``0.0``.
    use_student_t : bool
        Use Student-t NLL instead of Gaussian NLL.  Default: ``False``.
    student_t_nu : float
        Degrees of freedom for the Student-t loss.  Lower values give
        heavier tails.  Default: ``2.0``.
    var_kernel : int
        Kernel size for the ``'local_var'`` auxiliary loss.
        Default: ``3``.
    outlier_threshold : float
        Amplitude threshold (in units of the input standard deviation)
        for the ``'outlier'`` penalty.  Default: ``3.0``.
    """

    def __init__(self, model: Noise2Moments,
                 hard_logvar_floor: float = None):
        self.model             = model
        self.hard_logvar_floor = hard_logvar_floor

        self.loss_weights = {
            'tv':          0.0,
            'smoothness':  0.0,
            'sparsity':    0.0,
            'local_var':   0.0,
            'var_prior':   0.0,
            'outlier':     0.0,
            'positivity':  0.0,
        }

        self.use_student_t    = False
        self.student_t_nu     = 2.0
        self.var_kernel       = 3
        self.outlier_threshold = 3.0

    # --------------------------------------------------------- floor helper

    def _apply_floor(self, logvar: _torch.Tensor) -> _torch.Tensor:
        if self.hard_logvar_floor is not None:
            floor = _torch.tensor(self.hard_logvar_floor, device=logvar.device)
            logvar = _torch.maximum(logvar, floor)
        return logvar

    # ------------------------------------------------------------- losses

    def nll_loss(self, mean: _torch.Tensor,
                 logvar: _torch.Tensor,
                 y: _torch.Tensor) -> _torch.Tensor:
        """Heteroscedastic Gaussian NLL (up to constant)."""
        inv_var = _torch.exp(-logvar)
        return (0.5 * (inv_var * (y - mean) ** 2 + logvar)).mean()

    def student_t_nll(self, mean: _torch.Tensor,
                      logvar: _torch.Tensor,
                      y: _torch.Tensor,
                      nu: float = 2.0) -> _torch.Tensor:
        """Student-t NLL.

        Heavier tails than Gaussian: the residual penalty grows as
        log(1 + residual²/scale) rather than residual²/scale, down-weighting
        voxels with large disagreement between half-maps (flexible regions,
        outlier targets).  This produces a more conservative predicted mean
        and reduces the chance of extreme output values.
        """
        inv_var  = _torch.exp(-logvar)
        sq_error = (y - mean) ** 2
        # normalization constants depend only on nu — scalar computation
        log_norm = (
            _torch.lgamma(_torch.tensor((nu + 1) / 2, device=mean.device))
            - _torch.lgamma(_torch.tensor(nu / 2,     device=mean.device))
            - 0.5 * (_np.log(nu) + _np.log(_np.pi))
            - 0.5 * logvar
        )
        log_kernel = -0.5 * (nu + 1) * _torch.log1p(
            sq_error * inv_var / (nu + 1e-8)
        )
        return -(log_norm + log_kernel).mean()

    def _grad3d(self, x: _torch.Tensor):
        dz = x[..., 1:, :, :] - x[..., :-1, :, :]
        dy = x[..., :, 1:, :] - x[..., :, :-1, :]
        dx = x[..., :, :, 1:] - x[..., :, :, :-1]
        return dz, dy, dx

    def tv_loss(self, mean: _torch.Tensor) -> _torch.Tensor:
        """L1 total variation.  Each axis reduced separately (shapes differ)."""
        dz, dy, dx = self._grad3d(mean)
        return (dz.abs().mean() + dy.abs().mean() + dx.abs().mean()) / 3

    def smoothness_loss(self, mean: _torch.Tensor) -> _torch.Tensor:
        """L2 squared-gradient smoothness."""
        dz, dy, dx = self._grad3d(mean)
        return (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()) / 3

    def sparsity_loss(self, mean: _torch.Tensor) -> _torch.Tensor:
        """L1 sparsity: penalises all amplitudes proportionally to ``|mean|``."""
        return mean.abs().mean()

    def variance_prior_loss(self, logvar: _torch.Tensor,
                            prior_mean: float = -4.0,
                            prior_std:  float =  1.0) -> _torch.Tensor:
        """Gaussian prior on logvar; prevents runaway uncertainty."""
        return (0.5 * ((logvar - prior_mean) / prior_std) ** 2).mean()

    def outlier_penalty(self, mean: _torch.Tensor) -> _torch.Tensor:
        """Hinge squared penalty on predicted mean values beyond threshold.

        Zero for ``|mean| <= outlier_threshold``; grows quadratically for larger
        values.  Since the input is zero-mean unit-std, outlier_threshold=3.0
        covers the expected signal range and only penalises genuine outliers
        in the network output.  Unlike sparsity (L1), this does not penalise
        normal-amplitude signal at all.
        """
        excess = (mean.abs() - self.outlier_threshold).clamp(min=0.0)
        return excess.pow(2).mean()

    def positivity_loss(self, mean: _torch.Tensor) -> _torch.Tensor:
        """One-sided L1 hinge penalty that discourages negative mean values.

        Equal to relu(-mean).mean(): zero where mean >= 0, grows linearly
        below zero.  Softly pushes the predicted density to be non-negative
        without hard-constraining the network output.
        """
        return _F.relu(-mean).mean()

    def local_variance_penalty(self, mean: _torch.Tensor,
                               logvar: _torch.Tensor = None) -> _torch.Tensor:
        """Mean local spatial variance of the predicted volume.

        Penalises fine-grained noise in the output.  When `logvar` is
        provided the penalty is weighted by the predicted uncertainty
        exp(logvar), focusing smoothing pressure on uncertain (flexible)
        regions while leaving confident (rigid) regions unaffected.

        logvar is detached before computing the weights so that gradients
        flow only through `mean`; the noise model (logvar) is determined
        solely by the NLL term, avoiding circular feedback.  Weights are
        normalised to unit mean so that the effective strength of the
        loss_weight does not drift as the predicted uncertainty changes
        during training.
        """
        k    = self.var_kernel
        filt = _torch.ones(1, 1, k, k, k, device=mean.device) / (k ** 3)
        local_mean = _F.conv3d(mean, filt, padding='same')
        local_var  = _F.conv3d((mean - local_mean) ** 2, filt, padding='same')
        if logvar is not None:
            w         = logvar.detach().exp()
            w         = w / (w.mean() + 1e-8)
            local_var = local_var * w
        return local_var.mean()

    # -------------------------------------------------------------- train

    def train(self, data, lr: float, n_epoch: int, shuffle: bool = True):
        """Train for *n_epoch* epochs over *data*.

        Each sample is fed as a single-element batch (N=1);
        ``InstanceNorm3d`` in the model handles this correctly.

        Parameters
        ----------
        data : :class:`VolumePairs` or sequence of (tensor, tensor) pairs
            Training dataset.
        lr : float
            Adam learning rate.
        n_epoch : int
            Number of full passes over the dataset.
        shuffle : bool, optional
            If ``True``, randomise sample order each epoch and randomly
            swap ``(x, y)`` within each pair.  Swapping trains the network
            symmetrically on both half-maps, effectively doubling the
            training signal.  Default: ``True``.
        """
        optimizer = _torch.optim.Adam(self.model.parameters(), lr=lr)
        n         = len(data)
        self.model.train()

        for epoch in range(n_epoch):
            indices    = _torch.randperm(n).tolist() if shuffle else list(range(n))
            total_loss = 0.0

            for ix in indices:
                x, y = data[ix]
                if shuffle and _torch.rand(1).item() > 0.5:
                    x, y = y, x
                # add batch and channel dimensions → (1, 1, D, H, W)
                x = x[None, None].to(self.model.input_device())
                y = y[None, None].to(self.model.output_device())

                optimizer.zero_grad()

                mean, logvar = self.model(x)
                logvar       = self._apply_floor(logvar)

                if self.use_student_t:
                    loss = self.student_t_nll(mean, logvar, y, self.student_t_nu)
                else:
                    loss = self.nll_loss(mean, logvar, y)

                w = self.loss_weights
                if w['tv']          > 0: loss = loss + w['tv']          * self.tv_loss(mean)
                if w['smoothness']  > 0: loss = loss + w['smoothness']  * self.smoothness_loss(mean)
                if w['sparsity']    > 0: loss = loss + w['sparsity']    * self.sparsity_loss(mean)
                if w['local_var']   > 0: loss = loss + w['local_var']   * self.local_variance_penalty(mean, logvar)
                if w['var_prior']   > 0: loss = loss + w['var_prior']   * self.variance_prior_loss(logvar)
                if w['outlier']     > 0: loss = loss + w['outlier']     * self.outlier_penalty(mean)
                if w['positivity']  > 0: loss = loss + w['positivity']  * self.positivity_loss(mean)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1:>{len(str(n_epoch))}}/{n_epoch}"
                  f"  loss: {total_loss / n:.6f}")

        print("Training complete.")

    # ------------------------------------------------------------- predict

    @_torch.no_grad()
    def predict(self, x: _np.ndarray):
        """Return raw predicted mean and log-variance for a numpy volume.

        Parameters
        ----------
        x : numpy.ndarray
            Input volume (zero-mean, unit-std).

        Returns
        -------
        mean : torch.Tensor
            3-D CPU tensor; predicted denoised signal.
        logvar : torch.Tensor
            3-D CPU tensor; predicted log noise variance per voxel.
            Low values indicate confident (rigid) regions; high values
            indicate uncertain (flexible) regions.
        """
        self.model.eval()
        x_gpu        = _torch.from_numpy(x[None, None]).to(self.model.input_device())
        mean, logvar = self.model(x_gpu)
        logvar       = self._apply_floor(logvar)
        return mean[0, 0].cpu(), logvar[0, 0].cpu()

    @_torch.no_grad()
    def mace_step(self, x: _np.ndarray, alpha: float = 0.2) -> _np.ndarray:
        """Single MACE/PnP-ADMM consensus step: blend the STA reconstruction with the denoiser.

        Both *x* (the current STA estimate, unit-std) and the predicted
        *mean* are estimates of the same underlying signal with different
        noise levels.  The MMSE fused estimate weights each source by its
        precision (1/variance)::

            w_den(v) = exp(−logvar(v)) / (1 + exp(−logvar(v)))

            merged = x · (1 − α·w_den) + mean · (α·w_den)

        Behaviour by region:

        * **Rigid / well-resolved** — low logvar → w_den → 1 → denoiser dominates.
        * **Flexible / uncertain** — high logvar → w_den → 0 → reconstruction kept.
        * **Equal confidence** — logvar ≈ 0 → w_den = 0.5 → equal blend.

        The result is renormalised to zero-mean unit-std so that the next
        STA iteration and any subsequent denoiser call receive input in the
        expected scale.

        Parameters
        ----------
        x : numpy.ndarray
            Current STA estimate (zero-mean, unit-std).
        alpha : float, optional
            Global MACE step size in (0, 1].  Keep small (0.1–0.3) for
            stability; per-voxel *w_den* handles spatial adaptation.
            Default: ``0.2``.

        Returns
        -------
        numpy.ndarray
            Zero-mean unit-std volume ready for the next STA iteration.
        """
        mean, logvar = self.predict(x)
        mean   = mean.numpy()
        logvar = logvar.numpy()

        inv_var_den = _np.exp(-logvar)                        # 1/σ²_den, per-voxel
        w_den       = inv_var_den / (1.0 + inv_var_den)      # precision weight ∈ (0,1)

        merged = x * (1.0 - alpha * w_den) + mean * (alpha * w_den)

        # Renormalise: denoiser was trained on unit-std input; merged has
        # lower variance because noise was partially removed.
        merged = (merged - merged.mean()) / merged.std()
        return merged

    @_torch.no_grad()
    def denoise(self, x: _np.ndarray,
                use_confidence: bool = False,
                clip_sigma: float = 4.0,
                min_weight: float = 0.0,
                sharpness: float = 1.0,
                confidence_mode: str = 'sigmoid') -> _torch.Tensor:
        """Return the denoised map.

        Parameters
        ----------
        x : numpy.ndarray
            Input volume (zero-mean, unit-std).
        use_confidence : bool, optional
            If ``True``, multiply the predicted mean by a per-voxel
            confidence mask derived from the predicted log-variance.
            Default: ``False``.
        clip_sigma : float or None, optional
            Clip the raw network mean to ±*clip_sigma* before confidence
            weighting.  Since the input is unit-std, this keeps the output
            in a sensible range and prevents outlier values from appearing
            amplified when the background is suppressed.  ``None`` disables
            clipping.  Default: ``4.0``.
        min_weight : float, optional
            Floor for the confidence mask, in ``[0, 1)``.  Prevents
            uncertain regions from collapsing to exactly zero.  Only used
            when ``confidence_mode='minmax'``.  Default: ``0.0``.
        sharpness : float, optional
            Gamma exponent applied to the confidence mask.  ``1.0`` leaves
            the mask unchanged; values above 1 steepen the transition.
            Only used when ``confidence_mode='minmax'``.  Default: ``1.0``.
        confidence_mode : str, optional
            How to convert logvar to a ``[0, 1]`` confidence weight:

            ``'sigmoid'`` (default)
                ``w = exp(−logvar) / (1 + exp(−logvar))``.  Absolute
                precision-based weight consistent with :meth:`mace_step`.
                Voxels with low logvar (certain) get weight near 1; voxels
                with high logvar (uncertain) get weight near 0.  Does not
                suppress signal when the model is globally confident.
            ``'minmax'``
                ``w = 1 − normalize(logvar)``.  Relative ranking: the most
                uncertain voxel always gets weight 0 regardless of the
                absolute uncertainty level.  Useful for maximum contrast
                between rigid and flexible regions but can delete signal
                when all logvar values are low.

        Returns
        -------
        torch.Tensor
            Denoised 3-D volume (CPU tensor).
        """
        mean, logvar = self.predict(x)
        if clip_sigma is not None:
            mean = mean.clamp(-clip_sigma, clip_sigma)
        if use_confidence:
            if confidence_mode == 'sigmoid':
                inv_var    = _torch.exp(-logvar)
                confidence = inv_var / (1.0 + inv_var)
            elif confidence_mode == 'minmax':
                lv         = logvar
                confidence = 1.0 - (lv - lv.min()) / (lv.max() - lv.min() + 1e-8)
                if sharpness != 1.0:
                    confidence = confidence.pow(sharpness)
                confidence = min_weight + (1.0 - min_weight) * confidence
            else:
                raise ValueError(f"confidence_mode must be 'sigmoid' or 'minmax', got {confidence_mode!r}")
            mean = mean * confidence
        return mean
