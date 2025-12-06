import unittest

import torch

from models.v3.WaveletDownsampler import (
    WaveletDownsampler,
    WaveletDownsamplerModel,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestWaveletDownsampler(unittest.TestCase):
    def test_diff_channels(self):
        x = torch.randn(10, 20, 32, 64).to(device)
        downsampler = WaveletDownsampler(in_channels=x.shape[1], out_channels=40).to(
            device
        )
        y = downsampler(x)
        self.assertEqual(y.shape, (10, 40, 16, 32))

    def test_same_channels(self):
        x = torch.randn(10, 20, 32, 64).to(device)
        downsampler = WaveletDownsampler(in_channels=x.shape[1], out_channels=20).to(
            device
        )
        y = downsampler(x)
        self.assertEqual(y.shape, (10, 20, 16, 32))


class TestMambaHSIWaveletDownsampler(unittest.TestCase):
    def test_model(self):
        x = torch.randn(5, 16, 64, 64).to(device)
        model = WaveletDownsamplerModel(
            in_channels=16,
            hidden_dim=8,
            num_classes=10,
            use_residual=True,
            token_num=4,
            group_num=4,
            use_att=True,
        ).to(device)
        y = model(x)
        self.assertEqual(y.shape, (5, 10, 16, 16))


if __name__ == "__main__":
    unittest.main()
