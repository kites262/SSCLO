import torch
import torch.nn.functional as F
from torch import nn


class FourierLayer2D(nn.Module):
    """
    A layer that applies multiple Fourier Neural Operators to different channel groups
    """

    def __init__(
        self,
        channels_in,
        channels_out,
        channels_boundary_ratio=[0.3, 0.3, 0.2, 0.2],
        reserved_ratio=1.0,
        weights_size=(128, 64),  # (H, W)
        scale_factor: float = 0.5,
    ):
        super(FourierLayer2D, self).__init__()

        self.channels_boundary_ratio = channels_boundary_ratio

        self.fno_layers = nn.ModuleList()

        self.channels_in_boundary_index = [
            int(channels_in * ratio) for ratio in channels_boundary_ratio[:-1]
        ]
        self.channels_in_boundary_index.append(
            channels_in - sum(self.channels_in_boundary_index)
        )

        self.channels_out_boundary_index = [
            int(channels_out * ratio) for ratio in channels_boundary_ratio[:-1]
        ]
        self.channels_out_boundary_index.append(
            channels_out - sum(self.channels_out_boundary_index)
        )

        for i in range(len(self.channels_in_boundary_index)):
            fno_layer = FourierNeuralOperator2D(
                channels_in=self.channels_in_boundary_index[i],
                channels_out=self.channels_out_boundary_index[i],
                reserved_ratio=reserved_ratio,
                weights_size=weights_size,
                scale_factor=scale_factor,
            )
            self.fno_layers.append(fno_layer)

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        """

        x_splits = torch.split(
            x,
            self.channels_in_boundary_index,
            dim=1,
        )

        out_splits = []
        for i, x_split in enumerate(x_splits):
            out_split = self.fno_layers[i](x_split)
            out_splits.append(out_split)

        x_out = torch.cat(out_splits, dim=1)

        return x_out


class FourierNeuralOperator2D(nn.Module):
    """
    Fourier Neural Operator for 2D data
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        reserved_ratio: float = 0.5,
        weights_size=(512, 128),  # (H, W)
        scale_factor: float = 0.5,
    ):
        super(FourierNeuralOperator2D, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.reserved_ratio = reserved_ratio
        self.weight_size = weights_size
        self.scale_factor = scale_factor

        # 高频残差缩放参数
        self.high_freq_scale = nn.Parameter(torch.tensor(0.01))

        # 频域权重
        self.weights_real = nn.Parameter(
            torch.randn(channels_out, channels_in, weights_size[0], weights_size[1])
            * 0.01
        )
        self.weights_imag = nn.Parameter(
            torch.randn(channels_out, channels_in, weights_size[0], weights_size[1])
            * 0.01
        )

    def forward(self, x):
        B, _, H, W = x.shape

        m_h = int(H * self.reserved_ratio)
        m_w = int((W // 2 + 1) * self.reserved_ratio)

        weights = torch.complex(
            F.interpolate(
                self.weights_real, size=(m_h, m_w), mode="bilinear", align_corners=False
            ),
            F.interpolate(
                self.weights_imag, size=(m_h, m_w), mode="bilinear", align_corners=False
            ),
        )

        x_fft = torch.fft.rfft2(x, norm="ortho")
        x_fft_low = x_fft[:, :, :m_h, :m_w]

        # 频域线性映射
        out_fft_low = torch.einsum("ocij, bcij -> boij", weights, x_fft_low)

        # 构建输出频谱
        out_fft = torch.zeros(
            (B, self.channels_out, H, W // 2 + 1), dtype=torch.cfloat, device=x.device
        )

        # 高频残差保留
        if self.channels_in == self.channels_out:
            out_fft += self.high_freq_scale * x_fft
        else:
            out_fft += self.high_freq_scale * x_fft.mean(dim=1, keepdim=True)

        # 放回低频
        out_fft[:, :, :m_h, :m_w] = out_fft_low

        # iFFT
        x_out = torch.fft.irfft2(out_fft, s=(H, W), norm="ortho")

        # scale
        x_out = F.interpolate(
            x_out, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

        return x_out
