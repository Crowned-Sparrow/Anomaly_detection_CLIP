"""CPU-sized component checks; run on the server with PyTorch installed."""

import io
import unittest

import torch
from torch.nn import functional as F

from DAPO.adapter import LinearAdapter
from clip_ad.blocks.spectral import RadialSpectrum, SpectralFusion


class SpectralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.image = torch.randn(2, 3, 32, 32)
        self.tokens = [torch.randn(2, 17, 8) for _ in range(2)]

    def test_zero_gate_matches_baseline(self):
        baseline = LinearAdapter(8, 6, 2)
        spectral = SpectralFusion(6, 2, window_size=16, radial_bins=8)
        expected = baseline([x.clone() for x in self.tokens])
        actual = spectral([x.clone() for x in expected], self.image)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_finite_constant_image_and_frequency_order(self):
        spectrum = RadialSpectrum(window_size=16, radial_bins=8)
        constant_rgb = torch.ones(1, 3, 16, 16) * 0.5
        constant = (constant_rgb - spectrum.mean) / spectrum.std
        output = spectrum(constant, (2, 2))
        self.assertEqual(output.shape, (1, 4, 8))
        self.assertTrue(torch.isfinite(output).all())
        x = torch.arange(16).float()
        peaks = []
        for frequency in (1, 6):
            rgb = (0.5 + 0.4 * torch.cos(2 * torch.pi * frequency * x / 16)).view(1, 1, 1, 16).expand(1, 3, 16, 16)
            normalized = (rgb - spectrum.mean) / spectrum.std
            peaks.append(spectrum(normalized, (1, 1))[0, 0].argmax().item())
        self.assertGreater(peaks[1], peaks[0])

    def test_windows_match_patch_order_and_full_fft_pooling(self):
        spectrum = RadialSpectrum(window_size=8, radial_bins=6)
        actual = spectrum(self.image, (4, 4))
        rgb = self.image * spectrum.std + spectrum.mean
        gray = (rgb * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1)
        frequency = torch.fft.fftfreq(8)
        radius = torch.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
        bins = (radius / (0.5 ** 0.5) * 6).long().clamp(max=5)
        for row, col in ((0, 0), (1, 2), (3, 3)):
            window = gray[:, row * 8:(row + 1) * 8, col * 8:(col + 1) * 8]
            window = (window - window.mean((-2, -1), keepdim=True)) * spectrum.window
            full = torch.log1p(torch.fft.fft2(window, norm='ortho').abs())
            expected = torch.stack([
                full[:, bins == i].mean(-1) if (bins == i).any() else torch.zeros(2)
                for i in range(6)
            ], dim=-1)
            torch.testing.assert_close(actual[:, row * 4 + col], F.normalize(expected, dim=-1))

    def test_gate_and_encoder_receive_gradients(self):
        adapter = SpectralFusion(6, 2, window_size=16, radial_bins=8)
        target = [torch.randn(2, 16, 6) for _ in range(2)]
        def loss():
            outputs = adapter([x[:, 1:, :6].clone() for x in self.tokens], self.image)
            return sum(F.mse_loss(x, y) for x, y in zip(outputs, target))
        loss().backward()
        self.assertTrue(torch.isfinite(adapter.gates.grad).all())
        self.assertGreater(adapter.gates.grad.abs().sum().item(), 0)
        with torch.no_grad():
            adapter.gates.fill_(0.1)
        adapter.zero_grad()
        loss().backward()
        self.assertGreater(adapter.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(adapter.encoder[0].weight.grad).all())

    def test_checkpoint_roundtrip_preserves_config_and_predictions(self):
        adapter = SpectralFusion(6, 2, window_size=16, radial_bins=7, hidden_dim=11)
        with torch.no_grad():
            adapter.gates.fill_(0.2)
        buffer = io.BytesIO()
        torch.save(adapter.state_dict(), buffer)
        buffer.seek(0)
        restored = SpectralFusion(6, 2, **adapter.config)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertEqual(restored.config, adapter.config)
        expected = adapter([x[:, 1:, :6].clone() for x in self.tokens], self.image)
        actual = restored([x[:, 1:, :6].clone() for x in self.tokens], self.image)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_inputs_fail_explicitly(self):
        adapter = SpectralFusion(6, 2, window_size=16)
        with self.assertRaisesRegex(ValueError, 'input image'):
            adapter([x.clone() for x in self.tokens])
        with self.assertRaisesRegex(ValueError, 'square patch grid'):
            adapter([torch.randn(2, 15, 6), torch.randn(2, 15, 6)], self.image)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            RadialSpectrum()(self.image, (3, 3))


if __name__ == '__main__':
    unittest.main()
