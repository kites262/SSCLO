import torch
from torch import nn
from torch.nn import functional as F

from models.MambaHSI import BothMamba


class Model(nn.Module):
    def __init__(
        self,
        in_channels=128,
        hidden_dim=128,
        num_classes=10,
        use_residual=True,
        token_num=4,
        group_num=4,
        use_att=True,
    ):
        super().__init__()

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )

        self.mamba = nn.Sequential(
            BothMamba(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
            ),
            LocalContextEnhancer(
                channels=hidden_dim,
                expand_ratio=2,
                group_num=group_num,
            ),
            WaveletDownsampler(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            ),
            BothMamba(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
            ),
            LocalContextEnhancer(
                channels=hidden_dim,
                expand_ratio=2,
                group_num=group_num,
            ),
            WaveletDownsampler(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            ),
            BothMamba(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
            ),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
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


class WaveletDownsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.wavelet_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.GroupNorm(2, out_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_proj = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.gamma = nn.Parameter(1e-5 * torch.ones(1, out_channels, 1, 1))

    def _pad_input(self, x):
        _, _, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward(self, x):
        x = self._pad_input(x)
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # ll = (x00 + x01 + x10 + x11) / 2
        lh = (x00 - x01 + x10 - x11) / 2
        hl = (x00 + x01 - x10 - x11) / 2
        # hh = (x00 + x11 - x01 - x10) / 2

        x_wavelet = torch.cat([lh, hl], dim=1)
        x_wavelet = self.wavelet_proj(x_wavelet)

        x_avg = self.avg_pool(x)
        x_avg = self.avg_proj(x_avg)

        return x_avg + self.gamma * x_wavelet


class LocalContextEnhancer(nn.Module):
    """
    放置于 Mamba 与 Downsampler 之间的串联特征细化模块。
    结合了光谱混合（1x1 Conv）、局部空间上下文恢复（3x3 DWConv）与通道注意力。
    """

    def __init__(self, channels, expand_ratio=2, group_num=4):
        super(LocalContextEnhancer, self).__init__()
        hidden_dim = channels * expand_ratio

        # 1. 光谱特征混合与升维 (Spectral Mixing & Expansion)
        self.proj_in = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )

        # 2. 局部空间上下文恢复 (Spatial Context Restoration)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )

        # 3. 通道级特征重标定 (Channel Attention)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 4. 降维并投射回原通道 (Linear Projection)
        self.proj_out = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, channels),
        )

    def forward(self, x):
        identity = x

        x = self.proj_in(x)
        x = self.dwconv(x)
        x = x * self.channel_att(x)
        x = self.proj_out(x)

        return x + identity
