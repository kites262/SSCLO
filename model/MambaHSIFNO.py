import torch
from torch import nn

from model.MambaHSI import BothMamba


class MambaHSIFNO(nn.Module):
    def __init__(self, in_channels, embedding_channels=64, num_classes=10):
        super(MambaHSIFNO, self).__init__()

        self.in_channels = in_channels
        self.embedding_channels = embedding_channels
        self.num_classes = num_classes

        self.channels_boundary_ratio = [0.4, 0.3, 0.1, 0.2]
        token_num = 8
        use_residual = True
        group_num = 4
        use_att = True

        self.patch_embedding = nn.Sequential(
            FourierLayer2D(
                out_channels=embedding_channels,
                reserved_ratio=0.4,
                channels_boundary_ratio=self.channels_boundary_ratio,
            ),
            nn.GroupNorm(4, embedding_channels),
            nn.SiLU(),
        )

        self.mamba = nn.Sequential(
            BothMamba(
                channels=embedding_channels,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            BothMamba(
                channels=embedding_channels,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            BothMamba(
                channels=embedding_channels,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
            ),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_channels,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.mamba(x)

        logits = self.cls_head(x)
        return logits


class FourierLayer2D(nn.Module):
    """
    A layer that applies multiple Fourier Neural Operators to different channel groups
    """

    def __init__(
        self,
        out_channels,
        reserved_ratio=0.8,
        channels_boundary_ratio=[0.4, 0.3, 0.1, 0.2],
    ):
        super(FourierLayer2D, self).__init__()

        self.out_channels = out_channels
        self.reserved_ratio = reserved_ratio
        self.channels_boundary_ratio = channels_boundary_ratio
        self.modules_registered = False

        self.fno_layers = nn.ModuleList()

    def _init_modules(self, x):
        _, C, H, W = x.shape
        self.in_channels = C
        self.in_height = H
        self.in_width = W

        self.in_channels_index = [
            int(self.in_channels * ratio) for ratio in self.channels_boundary_ratio[:-1]
        ]
        self.in_channels_index.append(self.in_channels - sum(self.in_channels_index))

        self.out_channels_index = [
            int(self.out_channels * ratio)
            for ratio in self.channels_boundary_ratio[:-1]
        ]
        self.out_channels_index.append(self.out_channels - sum(self.out_channels_index))

        for i in range(len(self.in_channels_index)):
            fno_layer = FourierNeuralOperator2D(
                in_channels=self.in_channels_index[i],
                out_channels=self.out_channels_index[i],
                reserved_x=int((self.in_width // 2 + 1) * self.reserved_ratio),
                reserved_y=int(self.in_height * self.reserved_ratio),
            ).to(x.device)
            self.fno_layers.append(fno_layer)

        self.modules_registered = True

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        """

        if not self.modules_registered:
            self._init_modules(x)

        x_splits = torch.split(
            x,
            self.in_channels_index,
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
        in_channels: int,
        out_channels: int,
        reserved_x: int,
        reserved_y: int,
    ):
        super(FourierNeuralOperator2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reserved_x = reserved_x
        self.reserved_y = reserved_y

        # (C_out, C_in, m, m)
        self.weights_real = nn.Parameter(
            torch.randn(out_channels, in_channels, reserved_y, reserved_x) * 0.01
        )
        self.weights_imag = nn.Parameter(
            torch.randn(out_channels, in_channels, reserved_y, reserved_x) * 0.01
        )

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        """

        # 复数权重 (C_out,C_in,m,m)
        weights = torch.complex(self.weights_real, self.weights_imag)
        m_x = self.reserved_x
        m_y = self.reserved_y
        # weights = weights[:, :, :m_y, :m_x]

        B, _, H, W = x.shape

        # 对空间维做 2D FFT (B, C_in, H, W//2+1)
        x_fft = torch.fft.rfft2(x, norm="ortho")

        # 取低频子块 (B, C_in, m1, m2)
        x_fft_low = x_fft[:, :, :m_y, :m_x]

        # 频域投影 (B,C_in,m1,m2) x (C_out,C_in,m1,m2) -> (B,C_out,m1,m2)
        # print(f"Processing {weights.shape=} with {x_fft_low.shape=}")
        out_fft_low = torch.einsum("ocij, bcij -> boij", weights, x_fft_low)

        # 输出频谱 (B,C_out,H,W//2+1)
        out_fft = torch.zeros(
            (B, self.out_channels, H, W // 2 + 1),
            dtype=torch.cfloat,
            device=x.device,
        )

        # 放回频谱左上角
        out_fft[:, :, :m_y, :m_x] = out_fft_low

        # IFFT (B,C_in,H,W)
        x_out = torch.fft.irfft2(out_fft, s=(H, W), norm="ortho")

        return x_out
