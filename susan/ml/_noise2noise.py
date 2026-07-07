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

__all__ = ['Noise2Noise', 'Noise2NoiseTrainer']

import numpy as _np
import torch    as _torch
import torch.nn as _nn
import torch.nn.functional as _F


# ---------------------------------------------------------------------------
# Private backbone builders


def _make_convnet(n_feat: int, depth: int) -> _nn.ModuleList:
    """Build ConvNet backbone blocks (single-channel input)."""
    blocks = [_nn.Sequential(
        _nn.Conv3d(1, n_feat, 3, padding=1),
        _nn.ReLU(inplace=True),
    )]
    for _ in range(depth - 2):
        blocks.append(_nn.Sequential(
            _nn.Conv3d(n_feat, n_feat, 3, padding=1),
            _nn.InstanceNorm3d(n_feat, affine=True),
            _nn.ReLU(inplace=True),
        ))
    return blocks


def _make_unet(n_feat: int, levels: int) -> '_UNet3d':
    return _UNet3d(n_feat, levels)


class _UNet3d(_nn.Module):
    """Encoder-decoder with skip connections.  Single-channel input."""

    def __init__(self, n_feat: int, levels: int):
        super().__init__()
        self.levels = levels

        enc_ch    = []
        self.enc  = _nn.ModuleList()
        self.pool = _nn.ModuleList()
        in_ch = 1
        for l in range(levels):
            out_ch = n_feat * (2 ** l)
            self.enc.append(_nn.Sequential(
                _nn.Conv3d(in_ch, out_ch, 3, padding=1),
                _nn.InstanceNorm3d(out_ch, affine=True),
                _nn.ReLU(inplace=True),
            ))
            self.pool.append(_nn.MaxPool3d(2))
            enc_ch.append(out_ch)
            in_ch = out_ch

        bot_ch = n_feat * (2 ** levels)
        self.bottleneck = _nn.Sequential(
            _nn.Conv3d(in_ch, bot_ch, 3, padding=1),
            _nn.InstanceNorm3d(bot_ch, affine=True),
            _nn.ReLU(inplace=True),
        )
        in_ch = bot_ch

        self.up  = _nn.ModuleList()
        self.dec = _nn.ModuleList()
        for l in reversed(range(levels)):
            skip_ch = enc_ch[l]
            out_ch  = n_feat * (2 ** l)
            self.up.append(_nn.Upsample(scale_factor=2, mode='trilinear',
                                        align_corners=False))
            self.dec.append(_nn.Sequential(
                _nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1),
                _nn.InstanceNorm3d(out_ch, affine=True),
                _nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        self.out_feat = in_ch

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        skips = []
        for enc, pool in zip(self.enc, self.pool):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = _F.pad(x, [0, skip.shape[4] - x.shape[4],
                                0, skip.shape[3] - x.shape[3],
                                0, skip.shape[2] - x.shape[2]])
            x = dec(_torch.cat([x, skip], dim=1))
        return x


# ---------------------------------------------------------------------------


class Noise2Noise(_nn.Module):
    """Simple Noise2Noise denoiser for 3-D cryo-EM half-map pairs.

    .. warning:: **Experimental.**  This class is part of the machine-learning
       subpackage and may change or be removed in a future release.

    Takes one half-map as input and predicts the other as target.  The loss
    is either L2 (MSE) or L1 (MAE).  Backbone is either a sequential
    ConvNet or a UNet.

    Parameters
    ----------
    arch : str
        Backbone architecture: ``'convnet'`` or ``'unet'``.
        Default: ``'convnet'``.
    depth : int
        For ``'convnet'``: total number of blocks (≥ 2).
        For ``'unet'``: number of encoder/decoder levels (≥ 1).
        Default: ``6``.
    n_feat : int
        Feature channels (base channels for UNet).  Default: ``32``.
    gpus : list of int
        CUDA device indices.  Default: ``[0]``.
    """

    def __init__(self, arch: str = 'convnet', depth: int = 6,
                 n_feat: int = 32, gpus: list = [0]):
        super().__init__()

        assert arch in ('convnet', 'unet'), "arch must be 'convnet' or 'unet'"
        assert isinstance(depth,  int) and depth  >= 1
        assert isinstance(n_feat, int) and n_feat  > 0
        assert isinstance(gpus, (list, tuple)) and len(gpus) >= 1

        self.arch   = arch
        self.depth  = depth
        self.n_feat = n_feat
        self.gpus   = list(gpus)
        self.devices = [_torch.device(f'cuda:{i}') for i in self.gpus]

        if arch == 'convnet':
            blocks  = _make_convnet(n_feat, depth)
            n_blocks = len(blocks)
            n_parts  = len(self.devices)
            splits   = _torch.linspace(0, n_blocks, n_parts + 1, dtype=_torch.int32)
            self.parts = _nn.ModuleList()
            for i in range(n_parts):
                part_blocks = blocks[splits[i]:splits[i + 1]]
                assert len(part_blocks) > 0, (
                    f"Too many GPUs ({n_parts}) for depth ({depth}); "
                    f"need at most {n_blocks} GPUs"
                )
                self.parts.append(_nn.Sequential(*part_blocks).to(self.devices[i]))
            out_feat = n_feat
            self._unet = None

        else:  # unet — keep on devices[0]; no multi-GPU split
            self._unet = _UNet3d(n_feat, depth).to(self.devices[0])
            self.parts = _nn.ModuleList()
            out_feat   = self._unet.out_feat

        self.head = _nn.Conv3d(out_feat, 1, 3, padding=1).to(self.devices[-1])

    # ---------------------------------------------------------------- devices

    def input_device(self) -> _torch.device:
        """Device expected for network inputs."""
        return self.devices[0]

    def output_device(self) -> _torch.device:
        """Device on which the prediction is produced."""
        return self.devices[-1]

    # --------------------------------------------------------------- forward

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, 1, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted denoised volume, same shape, on :meth:`output_device`.
        """
        x = x.to(self.devices[0])
        if self._unet is not None:
            x = self._unet(x).to(self.devices[-1])
        else:
            for i, part in enumerate(self.parts):
                x = part(x)
                if i + 1 < len(self.parts):
                    x = x.to(self.devices[i + 1])
        return self.head(x)

    # -------------------------------------------------------------- utilities

    def reset_weights(self):
        """Re-initialise all learnable parameters."""
        def _reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(_reset)

    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        """Save architecture hyperparameters and weights."""
        _torch.save({
            'arch':       self.arch,
            'depth':      self.depth,
            'n_feat':     self.n_feat,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, gpus: list = [0]) -> 'Noise2Noise':
        """Reconstruct model from a file saved with :meth:`save`."""
        ckpt  = _torch.load(path, map_location='cpu', weights_only=True)
        model = cls(
            arch   = str(ckpt['arch']),
            depth  = int(ckpt['depth']),
            n_feat = int(ckpt['n_feat']),
            gpus   = gpus,
        )
        model.load_state_dict(ckpt['state_dict'])
        return model


# ---------------------------------------------------------------------------


class Noise2NoiseTrainer:
    """Training and inference wrapper for :class:`Noise2Noise`.

    .. warning:: **Experimental.**  This class is part of the machine-learning
       subpackage and may change or be removed in a future release.

    Implements Noise2Noise training: given a half-map pair ``(h1, h2)`` where
    both share the same underlying signal but have independent noise, the model
    learns to predict ``h2`` from ``h1`` (and vice versa).  At inference a
    single volume ``v`` is denoised by ``predict(v)``.

    Parameters
    ----------
    model : Noise2Noise
        Network to train.
    loss : str
        ``'l2'`` (MSE) or ``'l1'`` (MAE).  Default: ``'l2'``.
    """

    def __init__(self, model: Noise2Noise, loss: str = 'l2'):
        assert loss in ('l1', 'l2'), "loss must be 'l1' or 'l2'"
        self.model = model
        self.loss  = loss
        self._optimizer = None

    # ----------------------------------------------------------- optimizer

    def _get_optimizer(self, lr: float) -> _torch.optim.Adam:
        if self._optimizer is None:
            self._optimizer = _torch.optim.Adam(self.model.parameters(), lr=lr)
        return self._optimizer

    # ------------------------------------------------------------- loss fn

    def _loss_fn(self, pred: _torch.Tensor, target: _torch.Tensor) -> _torch.Tensor:
        if self.loss == 'l2':
            return _F.mse_loss(pred, target)
        return _F.l1_loss(pred, target)

    # -------------------------------------------------------------- train

    def train(self, vol_pairs, n_epochs: int = 10, lr: float = 1e-4,
              batch_size: int = 1, print_every: int = 10,
              directional: bool = False):
        """Train the network using Noise2Noise on half-map pairs.

        By default each sample ``(h1, h2)`` from *vol_pairs* contributes two
        symmetric terms, ``h1 → h2`` and ``h2 → h1`` — correct for plain
        denoising, where both halves share the same signal.

        With ``directional=True`` only the ``h1 → h2`` term is used: slot 0 is
        treated as the input and slot 1 as the target.  Use this for pairs
        produced by :meth:`VolumePairs.populate` with an angular perturbation,
        where slot 0 is a deliberately misaligned reconstruction and slot 1 is
        cleanly aligned; the reverse term would train the network to *introduce*
        misalignment and must be disabled.

        Parameters
        ----------
        vol_pairs : VolumePairs or iterable of (Tensor, Tensor)
            Source of half-map pairs.  Each element must be a pair of
            ``torch.Tensor`` with shape ``(1, D, H, W)`` or ``(D, H, W)``.
        n_epochs : int
            Number of passes over *vol_pairs*.  Default: ``10``.
        lr : float
            Learning rate.  Default: ``1e-4``.
        batch_size : int
            Number of pairs per gradient step.  Default: ``1``.
        print_every : int
            Print mean loss every this many epochs.  ``0`` disables printing.
            Default: ``10``.
        directional : bool
            When ``True``, train only ``slot 0 → slot 1`` (input → target)
            without the symmetric reverse term.  Default: ``False``.

        Returns
        -------
        list of float
            Mean loss per epoch.
        """
        opt    = self._get_optimizer(lr)
        dev_in = self.model.input_device()
        dev_out= self.model.output_device()

        self.model.train()
        epoch_losses = []

        for epoch in range(n_epochs):
            running = 0.0
            n_steps = 0
            opt.zero_grad()

            for step_idx, (h1, h2) in enumerate(vol_pairs):
                h1 = _ensure_5d(h1).to(dev_in)
                h2 = _ensure_5d(h2).to(dev_in)

                # h1 → h2  (slot 0 = input, slot 1 = target)
                pred_12 = self.model(h1)
                loss = self._loss_fn(pred_12, h2.to(dev_out))

                if not directional:
                    # h2 → h1 (symmetric term; only valid when both halves
                    # share the same signal, i.e. no directional perturbation).
                    pred_21 = self.model(h2)
                    loss = loss + self._loss_fn(pred_21, h1.to(dev_out))
                    loss = loss / 2.0

                loss.backward()
                running += loss.item()
                n_steps += 1

                if n_steps % batch_size == 0:
                    opt.step()
                    opt.zero_grad()

            if n_steps % batch_size != 0:
                opt.step()
                opt.zero_grad()

            mean_loss = running / max(n_steps, 1)
            epoch_losses.append(mean_loss)

            if print_every > 0 and (epoch + 1) % print_every == 0:
                print(f'  epoch {epoch + 1:>{len(str(n_epochs))}}/{n_epochs}'
                      f'  loss ({self.loss}) = {mean_loss:.6f}')

        self.model.eval()
        return epoch_losses

    # ---------------------------------------------------------- inference

    def predict(self, v: _np.ndarray) -> _np.ndarray:
        """Denoise a single volume.

        Parameters
        ----------
        v : numpy.ndarray
            Input volume, shape ``(D, H, W)``.

        Returns
        -------
        numpy.ndarray
            Denoised volume, same shape, float32.
        """
        self.model.eval()
        dev = self.model.input_device()
        with _torch.no_grad():
            x   = _torch.from_numpy(v.astype(_np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
            out = self.model(x)
        return out.squeeze().cpu().numpy()

    def denoise(self, v: _np.ndarray,
                patch_size: int = None,
                overlap: int = 8) -> _np.ndarray:
        """Denoise a volume, optionally with overlapping patches.

        When *patch_size* is ``None`` the whole volume is processed at once
        (same as :meth:`predict`).  Patch mode is useful when GPU memory is
        limited; overlapping regions are averaged.

        Parameters
        ----------
        v : numpy.ndarray
            Input volume, shape ``(D, H, W)``.
        patch_size : int or None
            Cubic patch side length.  ``None`` = whole volume.  Default: ``None``.
        overlap : int
            Overlap in voxels between adjacent patches.  Default: ``8``.

        Returns
        -------
        numpy.ndarray
            Denoised volume, same shape, float32.
        """
        if patch_size is None:
            return self.predict(v)

        self.model.eval()
        dev  = self.model.input_device()
        D, H, W = v.shape
        step = max(patch_size - overlap, 1)

        out = _np.zeros_like(v, dtype=_np.float32)
        cnt = _np.zeros_like(v, dtype=_np.float32)

        ranges = lambda n: range(0, max(n - patch_size, 0) + 1, step)

        with _torch.no_grad():
            for z in ranges(D):
                for y in ranges(H):
                    for x in ranges(W):
                        z1 = min(z + patch_size, D)
                        y1 = min(y + patch_size, H)
                        x1 = min(x + patch_size, W)
                        patch = v[z:z1, y:y1, x:x1].astype(_np.float32)
                        t = _torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(dev)
                        p = self.model(t).squeeze().cpu().numpy()
                        out[z:z1, y:y1, x:x1] += p
                        cnt[z:z1, y:y1, x:x1] += 1.0

        return out / _np.maximum(cnt, 1.0)

    def mace_step(self, v: _np.ndarray, alpha: float = 0.5) -> _np.ndarray:
        """MACE consensus step: blend input with network prediction.

        Returns ``alpha * predict(v) + (1 - alpha) * v``.

        Parameters
        ----------
        v : numpy.ndarray
            Input volume.
        alpha : float
            Blending weight (0 = no denoising, 1 = full prediction).
            Default: ``0.5``.

        Returns
        -------
        numpy.ndarray
            Blended volume, float32.
        """
        return alpha * self.predict(v) + (1.0 - alpha) * v.astype(_np.float32)

    # -------------------------------------------------------- persistence

    def save(self, path: str):
        """Save model weights and trainer state.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        _torch.save({
            'model_state': self.model.state_dict(),
            'model_arch':  self.model.arch,
            'model_depth': self.model.depth,
            'model_feat':  self.model.n_feat,
            'loss':        self.loss,
            'optimizer':   self._optimizer.state_dict() if self._optimizer else None,
        }, path)

    def load_state(self, path: str):
        """Restore weights and trainer state saved with :meth:`save`.

        Parameters
        ----------
        path : str
            Path to a file written by :meth:`save`.
        """
        ckpt = _torch.load(path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(ckpt['model_state'])
        self.loss = str(ckpt['loss'])
        if ckpt['optimizer'] is not None:
            opt = self._get_optimizer(1e-4)
            opt.load_state_dict(ckpt['optimizer'])


# ---------------------------------------------------------------------------
# Helper

def _ensure_5d(t: _torch.Tensor) -> _torch.Tensor:
    if t.dim() == 3:
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 4:
        return t.unsqueeze(0)
    return t
