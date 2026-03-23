import torch
import torch.nn.functional as F
from torch import nn

from models.MambaHSI import BothMamba


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


class MultiScaleConvBranch(nn.Module):
    """
    局部空间-光谱特征提取器。
    采用1x1卷积进行光谱维度的跨通道交互，结合3x3深度可分离卷积提取局部空间结构。
    """

    def __init__(self, channels, group_num=4):
        super(MultiScaleConvBranch, self).__init__()
        # 分支1: 1x1 卷积（提取细粒度光谱特征）
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, channels // 2),
            nn.SiLU(),
        )
        # 分支2: 3x3 深度可分离卷积（捕获局部空间上下文）
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels // 2,
                kernel_size=3,
                padding=1,
                groups=channels // 2,
                bias=False,
            ),
            nn.GroupNorm(group_num, channels // 2),
            nn.SiLU(),
        )
        # 局部特征聚合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.fusion(out)
        return out


class HybridMambaFusion(nn.Module):
    """
    双流网络融合模块。
    并联原始的 BothMamba（提取全局长序列特征）与 MultiScaleConvBranch（提取局部特征）。
    采用通道注意力（Channel Attention）机制进行自适应特征加权融合。
    """

    def __init__(
        self, channels, token_num, use_residual=True, group_num=4, use_att=True
    ):
        super(HybridMambaFusion, self).__init__()
        self.use_residual = use_residual

        # 全局分支: 传入 use_residual=False 避免在分支内部提前产生残差
        self.mamba_branch = BothMamba(
            channels=channels,
            token_num=token_num,
            use_residual=False,
            group_num=group_num,
            use_att=use_att,
        )

        # 局部分支
        self.conv_branch = MultiScaleConvBranch(channels, group_num)

        # 自适应特征融合 (通道注意力)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 通道降维
        self.out_proj = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 1. 并行特征提取
        x_global = self.mamba_branch(x)
        x_local = self.conv_branch(x)

        # 2. 特征拼接 (B, 2C, H, W)
        x_cat = torch.cat([x_global, x_local], dim=1)

        # 3. 通道注意力加权
        w = self.global_pool(x_cat)
        w = self.channel_att(w)
        x_fused = x_cat * w

        # 4. 降维并输出
        out = self.out_proj(x_fused)

        # 5. 整体残差连接
        if self.use_residual:
            return out + x
        return out


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
            HybridMambaFusion(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
            ),
            WaveletDownsampler(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            ),
            HybridMambaFusion(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
            ),
            WaveletDownsampler(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            ),
            HybridMambaFusion(
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
