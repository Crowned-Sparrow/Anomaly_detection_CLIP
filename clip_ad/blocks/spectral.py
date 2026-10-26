"""Image-window spectral residual for DAPO's adapted CLIP patch tokens."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RadialSpectrum(nn.Module):
    """Hann-windowed grayscale FFT magnitudes, averaged into radial bins.

    Input is the same CLIP-normalized, resized/cropped image used by the ViT.
    Windows have one center per patch, in row-major token order. Padding is
    replicated at image boundaries; no masks or target statistics are used.
    """

    def __init__(self, window_size=28, radial_bins=16):
        super().__init__()
        if window_size < 2 or radial_bins < 2:
            raise ValueError('window_size and radial_bins must be at least 2')
        self.window_size = window_size
        self.radial_bins = radial_bins
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1), persistent=False)
        hann = torch.hann_window(window_size, periodic=False)
        self.register_buffer('window', hann[:, None] * hann[None, :], persistent=False)
        fy = torch.fft.fftfreq(window_size)[:, None]
        fx = torch.fft.rfftfreq(window_size)[None, :]
        radius = torch.sqrt(fy.square() + fx.square())
        indices = (radius / math.sqrt(0.5) * radial_bins).long().clamp(max=radial_bins - 1).flatten()
        # rfft stores only one half-plane. Weight conjugate pairs so bin means
        # match radial pooling over the full 2D spectrum.
        weights = torch.full_like(radius, 2.0)
        weights[:, 0] = 1.0
        if window_size % 2 == 0:
            weights[:, -1] = 1.0
        weights = weights.flatten()
        counts = torch.zeros(radial_bins).scatter_add_(0, indices, weights)
        self.register_buffer('indices', indices, persistent=False)
        self.register_buffer('weights', weights, persistent=False)
        self.register_buffer('counts', counts.clamp_min(1), persistent=False)

    def forward(self, image, grid):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected CLIP-normalized RGB images [B, 3, H, W]')
        gh, gw = grid
        h, w = image.shape[-2:]
        if h % gh or w % gw:
            raise ValueError('Image dimensions must be divisible by the patch grid')
        sh, sw = h // gh, w // gw
        size = self.window_size
        if size < max(sh, sw):
            raise ValueError('Spectral window must be at least as large as a patch')
        # Always compute FFT in float32, including under mixed precision.
        with torch.autocast(device_type=image.device.type, enabled=False):
            rgb = image.float() * self.std + self.mean
            gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
            ph, pw = size - sh, size - sw
            gray = F.pad(gray, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode='replicate')
            windows = F.unfold(gray, kernel_size=size, stride=(sh, sw)).transpose(1, 2)
            windows = windows.reshape(image.shape[0], gh * gw, size, size)
            windows = (windows - windows.mean(dim=(-2, -1), keepdim=True)) * self.window
            magnitude = torch.log1p(torch.fft.rfft2(windows, norm='ortho').abs()).flatten(-2)
            bins = magnitude.new_zeros(*magnitude.shape[:-1], self.radial_bins)
            indices = self.indices.expand_as(magnitude)
            bins.scatter_add_(-1, indices, magnitude * self.weights)
            return F.normalize(bins / self.counts, dim=-1)


class SpectralFusion(nn.Module):
    """Fuse a spectral residual into already-adapted spatial tokens."""

    def __init__(self, dim_out, k, window_size=28, radial_bins=16, hidden_dim=128):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError('hidden_dim must be positive')
        self.config = dict(window_size=window_size, radial_bins=radial_bins, hidden_dim=hidden_dim)
        self.spectrum = RadialSpectrum(window_size, radial_bins)
        self.encoder = nn.Sequential(nn.Linear(radial_bins, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim_out))
        self.gates = nn.Parameter(torch.zeros(k))

    def forward(self, patch_tokens, image=None):
        if image is None:
            raise ValueError('Spectral fusion requires the preprocessed input image')
        spatial = patch_tokens
        if len(spatial) != len(self.gates):
            raise ValueError('Number of feature layers does not match spectral gates')
        descriptors = {}
        fused = []
        for i, tokens in enumerate(spatial):
            side = math.isqrt(tokens.shape[1])
            if side * side != tokens.shape[1]:
                raise ValueError('Spectral fusion currently requires a square patch grid')
            if side not in descriptors:
                descriptors[side] = F.normalize(self.encoder(self.spectrum(image, (side, side))), dim=-1)
            # Match spatial-token scale; the gate controls relative contribution.
            residual = descriptors[side] * tokens.norm(dim=-1, keepdim=True).detach()
            fused.append(tokens + self.gates[i].tanh() * residual)
        return fused
