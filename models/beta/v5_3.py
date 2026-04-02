import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch import Tensor, nn

from utils.Loss import resize

# =========================
# Helpers
# =========================


def _resolve_group_num(num_channels: int, preferred_groups: int = 4) -> int:
    """Return a valid group count for GroupNorm."""
    preferred_groups = max(1, int(preferred_groups))
    for g in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        group_num: int = 4,
        act: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.GroupNorm(_resolve_group_num(out_channels, group_num), out_channels),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SpectralBandDropout(nn.Module):
    """Randomly drops a small portion of spectral bands during training."""

    def __init__(self, drop_prob: float = 0.05):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        b, c, _, _ = x.shape
        keep_mask = (torch.rand(b, c, 1, 1, device=x.device) > self.drop_prob).to(
            x.dtype
        )
        return x * keep_mask


# =========================
# Baseline Mamba blocks
# =========================


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
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / self.token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(
                _resolve_group_num(self.channel_num, group_num), self.channel_num
            ),
            nn.SiLU(inplace=True),
        )

    def padding_feature(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if c < self.channel_num:
            pad_c = self.channel_num - c
            pad_features = torch.zeros((b, pad_c, h, w), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad_features], dim=1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        b, h, w, c_pad = x_pad.shape

        x_flat = x_pad.view(b * h * w, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(b, h, w, c_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)

        if self.use_residual:
            return x + x_proj[:, : x.shape[1], :, :]
        return x_proj[:, : x.shape[1], :, :]


class SpaMamba(nn.Module):
    def __init__(
        self,
        channels: int,
        use_residual: bool = True,
        group_num: int = 4,
        use_proj: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj

        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(_resolve_group_num(channels, group_num), channels),
                nn.SiLU(inplace=True),
            )

    def forward(self, x: Tensor) -> Tensor:
        x_re = x.permute(0, 2, 3, 1).contiguous()
        b, h, w, c = x_re.shape
        x_flat = x_re.view(b, -1, c)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(b, h, w, c)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)

        if self.use_residual:
            return x_recon + x
        return x_recon


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
        self.use_att = use_att
        self.use_residual = use_residual

        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(
            channels=channels,
            use_residual=use_residual,
            group_num=group_num,
        )
        self.spe_mamba = SpeMamba(
            channels=channels,
            token_num=token_num,
            use_residual=use_residual,
            group_num=group_num,
        )

    def forward(self, x: Tensor) -> Tensor:
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)

        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x

        if self.use_residual:
            return fusion_x + x
        return fusion_x


# =========================
# Your validated enhancement blocks
# =========================


class WaveletDownsampler(nn.Module):
    """
    Keep your current stable design:
    avg branch preserves low-frequency semantics;
    wavelet branch compensates for high-frequency local details.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_num: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.wavelet_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                _resolve_group_num(out_channels, min(group_num, 2)), out_channels
            ),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.gamma = nn.Parameter(1e-5 * torch.ones(1, out_channels, 1, 1))

    @staticmethod
    def _pad_input(x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._pad_input(x)
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        lh = (x00 - x01 + x10 - x11) / 2.0
        hl = (x00 + x01 - x10 - x11) / 2.0

        x_wavelet = torch.cat([lh, hl], dim=1)
        x_wavelet = self.wavelet_proj(x_wavelet)

        x_avg = self.avg_pool(x)
        x_avg = self.avg_proj(x_avg)
        return x_avg + self.gamma * x_wavelet


class LocalContextEnhancer(nn.Module):
    """
    Your current LCE, augmented only with DropPath.
    """

    def __init__(
        self,
        channels: int,
        expand_ratio: float = 2.0,
        group_num: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden_dim = max(channels, int(round(channels * expand_ratio)))

        self.proj_in = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_resolve_group_num(hidden_dim, group_num), hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.GroupNorm(_resolve_group_num(hidden_dim, group_num), hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_resolve_group_num(channels, group_num), channels),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.proj_in(x)
        x = self.dwconv(x)
        x = x * self.channel_att(x)
        x = self.proj_out(x)
        return identity + self.drop_path(x)


# =========================
# New low-risk spatial consistency head
# =========================


class BoundaryAwareLogitRefiner(nn.Module):
    """
    A tiny spatial-consistency head.
    It smooths logits inside homogeneous regions and preserves edges.
    """

    def __init__(
        self,
        feat_channels: int,
        group_num: int = 4,
        init_alpha: float = 1e-4,
    ):
        super().__init__()
        self.edge_head = nn.Sequential(
            nn.Conv2d(
                feat_channels,
                feat_channels,
                kernel_size=3,
                padding=1,
                groups=feat_channels,
                bias=False,
            ),
            nn.GroupNorm(_resolve_group_num(feat_channels, group_num), feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(feat_channels, 1, kernel_size=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, feat: Tensor, logits: Tensor) -> Tuple[Tensor, Tensor]:
        edge = torch.sigmoid(self.edge_head(feat))
        local_logits = F.avg_pool2d(logits, kernel_size=3, stride=1, padding=1)
        refined_logits = logits + self.alpha * (1.0 - edge) * (local_logits - logits)
        return refined_logits, edge


# =========================
# Final model
# =========================


class Model(nn.Module):
    """
    Recommended next-step model.

    Design principles:
    1) Keep your verified backbone: BothMamba + LCE + Wavelet.
    2) Add only low-risk regularization: spectral band dropout + DropPath.
    3) Add only one low-capacity spatial-consistency head at the output.
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
        spectral_drop_prob: float = 0.05,
        lce1_expand_ratio: float = 2.0,
        lce2_expand_ratio: float = 1.5,
        lce1_drop_path: float = 0.03,
        lce2_drop_path: float = 0.06,
        wavelet_dropout: float = 0.10,
        use_refiner: bool = True,
        refiner_init_alpha: float = 1e-4,
    ):
        super().__init__()
        self.use_refiner = use_refiner
        gn_hidden = _resolve_group_num(hidden_dim, group_num)

        self.input_regularizer = SpectralBandDropout(drop_prob=spectral_drop_prob)

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.GroupNorm(gn_hidden, hidden_dim),  # LayerNorm
            nn.SiLU(inplace=True),
        )

        self.stage1 = BothMamba(
            channels=hidden_dim,
            token_num=token_num,
            use_residual=use_residual,
            group_num=group_num,
            use_att=use_att,
        )
        self.lce1 = LocalContextEnhancer(
            channels=hidden_dim,
            expand_ratio=lce1_expand_ratio,
            group_num=group_num,
            drop_path=lce1_drop_path,
        )
        self.down1 = WaveletDownsampler(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            group_num=group_num,
            dropout=wavelet_dropout,
        )

        self.stage2 = BothMamba(
            channels=hidden_dim,
            token_num=token_num,
            use_residual=use_residual,
            group_num=group_num,
            use_att=use_att,
        )
        self.lce2 = LocalContextEnhancer(
            channels=hidden_dim,
            expand_ratio=lce2_expand_ratio,
            group_num=group_num,
            drop_path=lce2_drop_path,
        )
        self.down2 = WaveletDownsampler(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            group_num=group_num,
            dropout=wavelet_dropout,
        )

        self.stage3 = BothMamba(
            channels=hidden_dim,
            token_num=token_num,
            use_residual=use_residual,
            group_num=group_num,
            use_att=use_att,
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(_resolve_group_num(128, group_num), 128),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.use_refiner:
            self.refiner = BoundaryAwareLogitRefiner(
                feat_channels=hidden_dim,
                group_num=group_num,
                init_alpha=refiner_init_alpha,
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.input_regularizer(x)
        x = self.patch_embedding(x)

        x = self.stage1(x)
        x = self.lce1(x)
        x = self.down1(x)

        x = self.stage2(x)
        x = self.lce2(x)
        x = self.down2(x)

        x = self.stage3(x)
        return x

    def forward(
        self,
        x: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        feat = self.forward_features(x)
        logits = self.cls_head(feat)

        aux: Dict[str, Tensor] = {"feat": feat, "pre_refine_logits": logits}

        if self.use_refiner:
            logits, edge = self.refiner(feat, logits)
            aux["edge"] = edge

        if return_aux:
            return logits, aux
        return logits


# =========================
# Optional training utilities
# =========================


def total_variation_loss_from_logits(logits: Tensor) -> Tensor:
    """A light spatial smoothness regularizer for segmentation logits."""
    prob = torch.softmax(logits, dim=1)
    dh = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs().mean()
    dw = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs().mean()
    return dh + dw


def edge_target_from_mask(mask: Tensor) -> Tensor:
    """
    Build a simple binary edge target from label mask.
    Expected mask shape: [B, H, W] or [B, 1, H, W].
    """
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError("mask should have shape [B,H,W] or [B,1,H,W].")

    mask = mask.long()
    edge = torch.zeros_like(mask, dtype=torch.bool)

    edge[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    edge[:, :-1, :] |= mask[:, :-1, :] != mask[:, 1:, :]
    edge[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    edge[:, :, :-1] |= mask[:, :, :-1] != mask[:, :, 1:]

    return edge.unsqueeze(1).float()


def compute_loss(
    logits: Tensor,
    target: Tensor,
    aux: Optional[Dict[str, Tensor]] = None,
    label_smoothing: float = 0.1,
    tv_weight: float = 0.01,
    edge_weight: float = 0.03,
    ignore_index: int = -1,
) -> Tensor:
    """
    A ready-to-use loss function.
    For classification patches where target is a single label per sample,
    do not use this function directly.
    This is for dense segmentation-style supervision.
    """
    logits = resize(
        input=logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    ce = F.cross_entropy(
        logits,
        target.long(),
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )
    loss = ce

    if tv_weight > 0:
        ref_logits = aux.get("pre_refine_logits", logits) if aux is not None else logits
        ref_logits = resize(
            input=ref_logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = loss + tv_weight * total_variation_loss_from_logits(ref_logits)

    if aux is not None and ("edge" in aux) and edge_weight > 0:
        edge_pred = resize(
            input=aux["edge"],
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(1e-4, 1 - 1e-4)
        edge_target = edge_target_from_mask(target)
        loss = loss + edge_weight * F.binary_cross_entropy(edge_pred, edge_target)

    return loss


class LossCalculator:
    def __init__(
        self,
        label_smoothing: float = 0.1,
        tv_weight: float = 0.01,
        edge_weight: float = 0.03,
    ):
        self.label_smoothing = float(label_smoothing)
        self.tv_weight = float(tv_weight)
        self.edge_weight = float(edge_weight)

    def run_model(self, net, x: Tensor):
        return net(x, return_aux=True)

    def __call__(
        self,
        *,
        loss_func,
        model_output,
        label: Tensor,
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        del loss_func
        logits, aux = model_output
        total_loss = compute_loss(
            logits,
            label,
            aux=aux,
            label_smoothing=self.label_smoothing,
            tv_weight=self.tv_weight,
            edge_weight=self.edge_weight,
        )

        return {
            "loss": total_loss,
            "logits": logits,
            "aux": aux,
            "log_items": {
                "loss": total_loss,
            },
        }


if __name__ == "__main__":
    x = torch.randn(2, 128, 11, 11)
    model = Model(in_channels=128, hidden_dim=128, num_classes=10)
    logits, aux = model(x, return_aux=True)
    print("logits:", logits.shape)
    for k, v in aux.items():
        print(k, v.shape)
