import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


class SpeMamba(nn.Module):
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
                "mamba_ssm is required. Install via `pip install mamba-ssm`."
            )
        self.mamba = Mamba(
            d_model=self.group_channel_num, d_state=16, d_conv=4, expand=2
        )
        self.proj = nn.Sequential(nn.GroupNorm(group_num, self.channel_num), nn.SiLU())

    def padding_feature(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c < self.channel_num:
            pad_c = self.channel_num - c
            pad = torch.zeros((b, pad_c, h, w), dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = self.padding_feature(x).permute(0, 2, 3, 1).contiguous()
        b, h, w, c_pad = x_pad.shape
        x_seq = x_pad.view(b * h * w, self.token_num, self.group_channel_num)
        x_seq = self.mamba(x_seq)
        x_rec = x_seq.view(b, h, w, c_pad).permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_rec)
        x_proj = x_proj[:, : x.shape[1], :, :]
        return x + x_proj if self.use_residual else x_proj


class SpaMamba(nn.Module):
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
                "mamba_ssm is required. Install via `pip install mamba-ssm`."
            )
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        if self.use_proj:
            self.proj = nn.Sequential(nn.GroupNorm(group_num, channels), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.permute(0, 2, 3, 1).contiguous()
        b, h, w, c = x_seq.shape
        x_seq = x_seq.view(b, h * w, c)
        x_seq = self.mamba(x_seq)
        x_rec = x_seq.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_rec = self.proj(x_rec)
        return x_rec + x if self.use_residual else x_rec


class BothMamba(nn.Module):
    def __init__(
        self,
        channels: int,
        token_num: int,
        use_residual: bool,
        group_num: int = 4,
        use_att: bool = True,
    ):
        super().__init__()
        self.use_att = bool(use_att)
        self.use_residual = bool(use_residual)

        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(
            channels, use_residual=use_residual, group_num=group_num
        )
        self.spe_mamba = SpeMamba(
            channels,
            token_num=token_num,
            use_residual=use_residual,
            group_num=group_num,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        return fusion_x + x if self.use_residual else fusion_x


class SpectralSelectionGate(nn.Module):
    """Lightweight spectral selection inspired by SSC: reweight neighboring bands before heavy modeling."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.band_conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=pad, bias=True
        )
        self.expand = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x).squeeze(-1).transpose(1, 2)  # [B,1,C]
        y = self.band_conv(y).transpose(1, 2).unsqueeze(-1)  # [B,C,1,1]
        y = self.expand(y)
        return x * (1.0 + y)


class AdaptiveLocalContextEnhancer(nn.Module):
    """Multiscale local compensation: 1x1 mix + parallel 3x3/5x5 DWConv + channel attention."""

    def __init__(
        self,
        channels: int,
        expand_ratio: int = 2,
        group_num: int = 4,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden = channels * expand_ratio

        self.proj_in = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, hidden),
            nn.SiLU(),
        )
        self.dw3 = nn.Sequential(
            nn.Conv2d(
                hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False
            ),
            nn.GroupNorm(group_num, hidden),
            nn.SiLU(),
        )
        self.dw5 = nn.Sequential(
            nn.Conv2d(
                hidden, hidden, kernel_size=5, padding=2, groups=hidden, bias=False
            ),
            nn.GroupNorm(group_num, hidden),
            nn.SiLU(),
        )
        self.scale_mix = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, channels),
        )
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.proj_in(x)
        w = torch.softmax(self.scale_mix, dim=0)
        x = w[0] * self.dw3(x) + w[1] * self.dw5(x)
        x = x * self.channel_att(x)
        x = self.drop(x)
        x = self.proj_out(x)
        return x + identity


class CenterAwareReweight(nn.Module):
    """Inject a center prior without changing sequence construction."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hs = max(1, h // 4)
        ws = max(1, w // 4)
        h0 = max(0, h // 2 - hs // 2)
        w0 = max(0, w // 2 - ws // 2)
        center = x[:, :, h0 : h0 + hs, w0 : w0 + ws].mean(dim=(2, 3), keepdim=True)
        global_feat = x.mean(dim=(2, 3), keepdim=True)
        gate = self.mlp(torch.cat([center.expand_as(global_feat), global_feat], dim=1))
        return x * (1.0 + gate)


class WaveletDownsamplerV2(nn.Module):
    """Avg low-frequency path + learnable wavelet high-frequency compensation."""

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
        band_num = 3 if self.use_hh else 2

        self.band_logits = nn.Parameter(torch.zeros(band_num))
        self.wavelet_proj = nn.Sequential(
            nn.Conv2d(in_channels * band_num, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, out_channels),
            nn.SiLU(),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.gamma = nn.Parameter(gamma_init * torch.ones(1, out_channels, 1, 1))

    @staticmethod
    def _pad_input(x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_input(x)
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5

        band_weights = torch.softmax(self.band_logits, dim=0)
        bands = [band_weights[0] * lh, band_weights[1] * hl]
        if self.use_hh:
            hh = (x00 - x01 - x10 + x11) * 0.5
            bands.append(band_weights[2] * hh)

        x_wave = self.wavelet_proj(torch.cat(bands, dim=1))
        x_avg = self.avg_proj(self.avg_pool(x))
        return x_avg + self.gamma * x_wave


class SpatialConsistencyRefiner(nn.Module):
    """Edge-gated local diffusion on logits to suppress island-like misclassification."""

    def __init__(
        self,
        num_classes: int,
        feat_channels: int,
        iters: int = 2,
        max_step: float = 0.30,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.iters = int(iters)
        self.max_step = float(max_step)

        self.edge_head = nn.Sequential(
            nn.Conv2d(
                feat_channels,
                feat_channels,
                kernel_size=3,
                padding=1,
                groups=max(1, feat_channels // 16),
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv2d(feat_channels, 1, kernel_size=1, bias=True),
        )
        self.step_logit = nn.Parameter(torch.tensor(0.0))

        kernel = (
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
            )
            / 4.0
        )
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3), persistent=False)

    def forward(
        self, logits: torch.Tensor, feat: torch.Tensor, return_edge: bool = False
    ):
        probs = torch.softmax(logits, dim=1)
        edge = torch.sigmoid(self.edge_head(feat))
        smooth_gate = 1.0 - edge
        step = torch.sigmoid(self.step_logit) * self.max_step

        weight = self.kernel.repeat(self.num_classes, 1, 1, 1)
        for _ in range(self.iters):
            neigh = F.conv2d(probs, weight=weight, padding=1, groups=self.num_classes)
            probs = probs * (1.0 - step * smooth_gate) + neigh * (step * smooth_gate)

        refined = torch.log(probs.clamp_min(1e-6))
        if return_edge:
            return refined, edge
        return refined


class Model(nn.Module):
    """
    Iterative optimized version on top of the user's current best model.
    Keeps:
      PatchEmbedding -> BothMamba -> Local enhancement -> Wavelet downsampling -> ...
    Adds:
      1) SpectralSelectionGate
      2) Multiscale local compensation
      3) CenterAwareReweight
      4) HH-enabled wavelet downsampling
      5) SpatialConsistencyRefiner on the output logits
    Output shape is kept consistent with the current implementation.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 10,
        use_residual: bool = True,
        token_num: int = 4,
        group_num: int = 4,
        use_att: bool = True,
        use_spectral_select: bool = True,
        use_center_gate: bool = True,
        use_refiner: bool = True,
        refiner_iters: int = 2,
    ):
        super().__init__()
        self.use_spectral_select = bool(use_spectral_select)
        self.use_center_gate = bool(use_center_gate)
        self.use_refiner = bool(use_refiner)

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )
        if self.use_spectral_select:
            self.spectral_select = SpectralSelectionGate(hidden_dim, kernel_size=7)

        self.stage1_mamba = BothMamba(
            hidden_dim, token_num, use_residual, group_num=group_num, use_att=use_att
        )
        self.stage1_lce = AdaptiveLocalContextEnhancer(
            hidden_dim, expand_ratio=2, group_num=group_num
        )
        if self.use_center_gate:
            self.stage1_center = CenterAwareReweight(hidden_dim)
        self.down1 = WaveletDownsamplerV2(
            hidden_dim, hidden_dim, group_num=group_num, use_hh=True
        )

        self.stage2_mamba = BothMamba(
            hidden_dim, token_num, use_residual, group_num=group_num, use_att=use_att
        )
        self.stage2_lce = AdaptiveLocalContextEnhancer(
            hidden_dim, expand_ratio=2, group_num=group_num
        )
        if self.use_center_gate:
            self.stage2_center = CenterAwareReweight(hidden_dim)
        self.down2 = WaveletDownsamplerV2(
            hidden_dim, hidden_dim, group_num=group_num, use_hh=True
        )

        self.stage3_mamba = BothMamba(
            hidden_dim, token_num, use_residual, group_num=group_num, use_att=use_att
        )
        self.stage3_lce = AdaptiveLocalContextEnhancer(
            hidden_dim, expand_ratio=2, group_num=group_num
        )
        if self.use_center_gate:
            self.stage3_center = CenterAwareReweight(hidden_dim)

        self.pre_head = nn.Sequential(
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
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.use_refiner:
            self.refiner = SpatialConsistencyRefiner(
                num_classes, hidden_dim, iters=refiner_iters, max_step=0.30
            )

    def _apply_stage(
        self,
        x: torch.Tensor,
        mamba: nn.Module,
        lce: nn.Module,
        center_gate: Optional[nn.Module],
    ) -> torch.Tensor:
        x = mamba(x)
        x = lce(x)
        if center_gate is not None:
            x = center_gate(x)
        return x

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = self.patch_embedding(x)
        if self.use_spectral_select:
            x = self.spectral_select(x)

        x = self._apply_stage(
            x, self.stage1_mamba, self.stage1_lce, getattr(self, "stage1_center", None)
        )
        x = self.down1(x)

        x = self._apply_stage(
            x, self.stage2_mamba, self.stage2_lce, getattr(self, "stage2_center", None)
        )
        x = self.down2(x)

        feat = self._apply_stage(
            x, self.stage3_mamba, self.stage3_lce, getattr(self, "stage3_center", None)
        )
        feat = self.pre_head(feat)

        logits = self.cls_head(feat)
        aux: Dict[str, torch.Tensor] = {"pre_refine_logits": logits}

        if self.use_refiner:
            logits, edge = self.refiner(logits, feat, return_edge=True)
            aux["edge"] = edge

        if return_aux:
            aux["logits"] = logits
            aux["feat"] = feat
            return aux
        return logits


def total_variation_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    dh = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
    dw = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
    return dh + dw


def edge_target_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Build a simple binary boundary target from segmentation labels.
    mask: [B,H,W] or [B,1,H,W], integer labels.
    """
    if mask.dim() == 4:
        mask = mask[:, 0]
    mask = mask.long()
    dh = (mask[:, 1:, :] != mask[:, :-1, :]).float()
    dw = (mask[:, :, 1:] != mask[:, :, :-1]).float()

    edge = torch.zeros(
        (mask.shape[0], mask.shape[1], mask.shape[2]),
        device=mask.device,
        dtype=torch.float32,
    )
    edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], dh)
    edge[:, :-1, :] = torch.maximum(edge[:, :-1, :], dh)
    edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], dw)
    edge[:, :, :-1] = torch.maximum(edge[:, :, :-1], dw)
    return edge.unsqueeze(1)


if __name__ == "__main__":
    model = Model(in_channels=128, hidden_dim=128, num_classes=10)
    x = torch.randn(2, 128, 16, 16)
    with torch.no_grad():
        y = model(x)
    print("output shape:", tuple(y.shape))
