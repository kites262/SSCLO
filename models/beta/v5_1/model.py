import math
from typing import Optional

import torch
from mamba_ssm import Mamba
from torch import nn
from torch.nn import functional as F


class SpeMamba(nn.Module):
    """
    Spectral (channel-group) Mamba:
    For each spatial position, run a short sequence (length=token_num) to model spectral deps.
    """

    def __init__(
        self,
        channels: int,
        token_num: int = 8,
        use_residual: bool = True,
        group_num: int = 4,
    ):
        super().__init__()
        self.token_num = int(token_num)
        self.use_residual = bool(use_residual)

        self.group_channel_num = math.ceil(channels / self.token_num)
        self.channel_num = self.token_num * self.group_channel_num

        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required. Install via: pip install mamba-ssm"
            )

        self.mamba = Mamba(
            d_model=self.group_channel_num, d_state=16, d_conv=4, expand=2
        )
        self.proj = nn.Sequential(nn.GroupNorm(group_num, self.channel_num), nn.SiLU())

    def _pad_channels(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c == self.channel_num:
            return x
        if c > self.channel_num:
            return x[:, : self.channel_num, :, :]
        pad_c = self.channel_num - c
        pad = torch.zeros((b, pad_c, h, w), dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = self._pad_channels(x)  # [B, Cpad, H, W]
        x_seq = x_pad.permute(0, 2, 3, 1).contiguous()  # [B, H, W, Cpad]
        b, h, w, cpad = x_seq.shape
        x_seq = x_seq.view(b * h * w, self.token_num, self.group_channel_num)
        x_seq = self.mamba(x_seq)
        x_rec = x_seq.view(b, h, w, cpad).permute(0, 3, 1, 2).contiguous()
        x_rec = self.proj(x_rec)

        if self.use_residual:
            if x_rec.shape[1] != x.shape[1]:
                x_rec = x_rec[:, : x.shape[1], :, :]
            return x + x_rec
        return x_rec


class SpaMamba(nn.Module):
    """
    Baseline spatial Mamba: flatten H*W into one 1D sequence (raster scan).
    """

    def __init__(
        self,
        channels: int,
        use_residual: bool = True,
        group_num: int = 4,
        use_proj: bool = True,
    ):
        super().__init__()
        self.use_residual = bool(use_residual)
        self.use_proj = bool(use_proj)

        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required. Install via: pip install mamba-ssm"
            )

        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        if self.use_proj:
            self.proj = nn.Sequential(nn.GroupNorm(group_num, channels), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_re = x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        b, h, w, c = x_re.shape
        x_seq = x_re.view(b, h * w, c)
        x_seq = self.mamba(x_seq)
        x_rec = x_seq.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_rec = self.proj(x_rec)
        return x_rec + x if self.use_residual else x_rec


class AxialSpaMamba(nn.Module):
    """
    Axial spatial Mamba: run Mamba along rows and columns (optionally bidirectional).
    This often improves spatial coherence vs raster flatten.
    """

    def __init__(
        self,
        channels: int,
        use_residual: bool = True,
        group_num: int = 4,
        use_proj: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.use_residual = bool(use_residual)
        self.use_proj = bool(use_proj)
        self.bidirectional = bool(bidirectional)

        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required. Install via: pip install mamba-ssm"
            )

        self.mamba_row = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.mamba_col = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)

        if self.use_proj:
            self.proj = nn.Sequential(nn.GroupNorm(group_num, channels), nn.SiLU())

    def _run_1d(self, m: nn.Module, seq: torch.Tensor) -> torch.Tensor:
        if not self.bidirectional:
            return m(seq)
        out_fwd = m(seq)
        out_rev = m(torch.flip(seq, dims=[1]))
        out_rev = torch.flip(out_rev, dims=[1])
        return 0.5 * (out_fwd + out_rev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Row-wise: [B*H, W, C]
        x_row = x.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
        x_row = self._run_1d(self.mamba_row, x_row)
        x_row = x_row.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # Col-wise: [B*W, H, C]
        x_col = x.permute(0, 3, 2, 1).contiguous().view(b * w, h, c)
        x_col = self._run_1d(self.mamba_col, x_col)
        x_col = x_col.view(b, w, h, c).permute(0, 3, 2, 1).contiguous()

        x_rec = 0.5 * (x_row + x_col)
        if self.use_proj:
            x_rec = self.proj(x_rec)
        return x_rec + x if self.use_residual else x_rec


class BothMamba(nn.Module):
    """
    Spectral-Spatial fusion block.

    sub_residual=False avoids 'double residual amplification' seen in some baseline variants.
    """

    def __init__(
        self,
        channels: int,
        token_num: int,
        use_residual: bool,
        group_num: int = 4,
        use_att: bool = True,
        spatial_mode: str = "flat",  # "flat" | "axial"
        sub_residual: bool = False,  # residual inside spa/spe
    ):
        super().__init__()
        self.use_att = bool(use_att)
        self.use_residual = bool(use_residual)

        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        if spatial_mode == "axial":
            self.spa = AxialSpaMamba(
                channels,
                use_residual=sub_residual,
                group_num=group_num,
                bidirectional=True,
            )
        elif spatial_mode == "flat":
            self.spa = SpaMamba(
                channels, use_residual=sub_residual, group_num=group_num
            )
        else:
            raise ValueError(f"Unknown spatial_mode={spatial_mode}")

        self.spe = SpeMamba(
            channels,
            token_num=token_num,
            use_residual=sub_residual,
            group_num=group_num,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spa_x = self.spa(x)
        spe_x = self.spe(x)
        if self.use_att:
            w = self.softmax(self.weights)
            fusion = spa_x * w[0] + spe_x * w[1]
        else:
            fusion = spa_x + spe_x
        return fusion + x if self.use_residual else fusion


class LocalContextEnhancer(nn.Module):
    """
    1x1 expand + 3x3 DWConv + channel attention + 1x1 project + residual.
    """

    def __init__(
        self,
        channels: int,
        expand_ratio: int = 2,
        group_num: int = 4,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden = channels * int(expand_ratio)

        self.proj_in = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, hidden),
            nn.SiLU(),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False
            ),
            nn.GroupNorm(group_num, hidden),
            nn.SiLU(),
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, channels),
        )
        self.drop = nn.Dropout2d(drop) if drop and drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.proj_in(x)
        x = self.dwconv(x)
        x = x * self.channel_att(x)
        x = self.drop(x)
        x = self.proj_out(x)
        return x + identity


class WaveletDownsamplerV2(nn.Module):
    """
    Haar-like wavelet downsampling:
    - low-frequency: avgpool branch
    - high-frequency: LH/HL/HH bands (configurable) + learnable gamma (small init)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_num: int = 4,
        use_hh: bool = True,
        drop: float = 0.1,
        gamma_init: float = 1e-5,
    ):
        super().__init__()
        self.use_hh = bool(use_hh)
        bands = 3 if self.use_hh else 2

        self.wavelet_proj = nn.Sequential(
            nn.Conv2d(in_channels * bands, out_channels, 1, bias=False),
            nn.GroupNorm(group_num, out_channels),
            nn.SiLU(),
            nn.Dropout(drop) if drop and drop > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_proj = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.gamma = nn.Parameter(gamma_init * torch.ones(1, out_channels, 1, 1))

    @staticmethod
    def _pad_even(x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_even(x)

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        if self.use_hh:
            hh = (x00 - x01 - x10 + x11) * 0.5
            x_wave = torch.cat([lh, hl, hh], dim=1)
        else:
            x_wave = torch.cat([lh, hl], dim=1)

        x_wave = self.wavelet_proj(x_wave)
        x_avg = self.avg_proj(self.avg_pool(x))
        return x_avg + self.gamma * x_wave


class UpFuseBlock(nn.Module):
    """
    upsample -> concat skip -> 1x1 mix -> DWConv -> channel attention
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, group_num: int = 4):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.GroupNorm(group_num, out_ch),
            nn.SiLU(),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.GroupNorm(group_num, out_ch),
            nn.SiLU(),
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[-2:] != skip.shape[-2:]:
            h = min(x.shape[-2], skip.shape[-2])
            w = min(x.shape[-1], skip.shape[-1])
            x = x[..., :h, :w]
            skip = skip[..., :h, :w]

        x = torch.cat([x, skip], dim=1)
        x = self.mix(x)
        x = self.dw(x)
        x = x * self.att(x)
        return x


class EdgeAwareSpatialRefiner(nn.Module):
    """
    Edge-aware diffusion on class probabilities to reduce salt-and-pepper noise.
    """

    def __init__(
        self,
        num_classes: int,
        feat_channels: int,
        iters: int = 3,
        max_alpha: float = 0.35,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.iters = int(iters)
        self.max_alpha = float(max_alpha)

        self.edge_pred = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(feat_channels, 1, 1, bias=True),
        )
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

        k = (
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
            )
            / 4.0
        )
        self.register_buffer("kernel", k.view(1, 1, 3, 3), persistent=False)

    def forward(self, logits: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        edge = torch.sigmoid(self.edge_pred(feat))  # [B,1,H,W]
        gate = 1.0 - edge  # inside~1, boundary~0
        alpha = torch.sigmoid(self.log_alpha) * self.max_alpha

        weight = self.kernel.repeat(self.num_classes, 1, 1, 1)
        for _ in range(self.iters):
            neigh = F.conv2d(probs, weight=weight, padding=1, groups=self.num_classes)
            probs = probs * (1.0 - alpha * gate) + neigh * (alpha * gate)

        eps = 1e-6
        return torch.log(probs.clamp(min=eps))


class Model(nn.Module):
    """
    Proposed v5:
    Encoder: BothMamba + LCE + WaveletDownsampler (twice)
    Decoder: 2x upsample with skip connections
    Refine:  edge-aware diffusion on logits (optional)
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 10,
        token_num: int = 4,
        group_num: int = 4,
        use_att: bool = True,
        spatial_mode: str = "axial",  # "flat" aligns closer to baseline; "axial" often better for coherence
        drop_lce: float = 0.0,
        use_refiner: bool = True,
        refiner_iters: int = 3,
    ):
        super().__init__()
        self.use_refiner = bool(use_refiner)

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )

        self.enc1 = nn.Sequential(
            BothMamba(
                hidden_dim,
                token_num,
                use_residual=True,
                group_num=group_num,
                use_att=use_att,
                spatial_mode=spatial_mode,
                sub_residual=False,
            ),
            LocalContextEnhancer(
                hidden_dim, expand_ratio=2, group_num=group_num, drop=drop_lce
            ),
        )
        self.down1 = WaveletDownsamplerV2(
            hidden_dim,
            hidden_dim,
            group_num=group_num,
            use_hh=True,
            drop=0.1,
            gamma_init=1e-5,
        )

        self.enc2 = nn.Sequential(
            BothMamba(
                hidden_dim,
                token_num,
                use_residual=True,
                group_num=group_num,
                use_att=use_att,
                spatial_mode=spatial_mode,
                sub_residual=False,
            ),
            LocalContextEnhancer(
                hidden_dim, expand_ratio=2, group_num=group_num, drop=drop_lce
            ),
        )
        self.down2 = WaveletDownsamplerV2(
            hidden_dim,
            hidden_dim,
            group_num=group_num,
            use_hh=True,
            drop=0.1,
            gamma_init=1e-5,
        )

        self.bottleneck = nn.Sequential(
            BothMamba(
                hidden_dim,
                token_num,
                use_residual=True,
                group_num=group_num,
                use_att=use_att,
                spatial_mode=spatial_mode,
                sub_residual=False,
            ),
            LocalContextEnhancer(
                hidden_dim, expand_ratio=2, group_num=group_num, drop=drop_lce
            ),
        )

        self.up1 = UpFuseBlock(hidden_dim, hidden_dim, hidden_dim, group_num=group_num)
        self.up2 = UpFuseBlock(hidden_dim, hidden_dim, hidden_dim, group_num=group_num)

        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, num_classes, 1, bias=True),
        )

        if self.use_refiner:
            self.refiner = EdgeAwareSpatialRefiner(
                num_classes, hidden_dim, iters=refiner_iters, max_alpha=0.35
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.patch_embedding(x)

        s1 = self.enc1(x0)
        x1 = self.down1(s1)

        s2 = self.enc2(x1)
        x2 = self.down2(s2)

        b = self.bottleneck(x2)

        d1 = self.up1(b, s2)
        d2 = self.up2(d1, s1)

        logits = self.head(d2)
        if self.use_refiner:
            logits = self.refiner(logits, d2)
        return logits


def total_variation_loss(probs: torch.Tensor) -> torch.Tensor:
    """
    TV loss on probability maps: encourages piecewise smooth predictions.
    probs: [B,K,H,W] after softmax.
    """
    dh = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
    dw = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
    return dh + dw
