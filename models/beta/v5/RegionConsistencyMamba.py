import torch
from torch import nn
from torch.nn import functional as F

from models.MambaHSI import BothMamba
from models.v4.LocalContextEnhancer import LocalContextEnhancer, WaveletDownsampler


class RegionConsistencyRefiner(nn.Module):
    """
    Edge-aware local consensus on logits.
    Smooths homogeneous regions while preserving boundaries via a learned gate.
    """

    def __init__(
        self,
        guidance_channels,
        num_classes,
        group_num=4,
        kernel_sizes=(3, 5),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_sizes = tuple(kernel_sizes)

        self.boundary_predictor = nn.Sequential(
            nn.GroupNorm(group_num, guidance_channels),
            nn.Conv2d(
                guidance_channels,
                guidance_channels,
                kernel_size=3,
                padding=1,
                groups=guidance_channels,
                bias=False,
            ),
            nn.GroupNorm(group_num, guidance_channels),
            nn.SiLU(),
            nn.Conv2d(guidance_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.boundary_predictor[-1].bias, -1.5)

        # Start from gentle smoothing and let training decide how much to use.
        self.mix_logits = nn.Parameter(torch.full((len(self.kernel_sizes),), -2.0))

        for kernel_size in self.kernel_sizes:
            weight = torch.full(
                (num_classes, 1, kernel_size, kernel_size),
                1.0 / (kernel_size * kernel_size),
            )
            self.register_buffer(f"kernel_{kernel_size}", weight)

    def _smooth_logits(self, logits, kernel_size):
        pad = kernel_size // 2
        kernel = getattr(self, f"kernel_{kernel_size}").to(dtype=logits.dtype)
        logits = F.pad(logits, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(logits, kernel, groups=self.num_classes)

    def forward(self, guidance, logits):
        boundary_prob = torch.sigmoid(self.boundary_predictor(guidance))
        region_gate = 1.0 - boundary_prob

        refined_logits = logits
        for mix_logit, kernel_size in zip(self.mix_logits, self.kernel_sizes):
            smoothed_logits = self._smooth_logits(refined_logits, kernel_size)
            mix = torch.sigmoid(mix_logit)
            refined_logits = refined_logits + mix * region_gate * (
                smoothed_logits - refined_logits
            )

        return refined_logits


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
        refine_kernel_sizes=(3, 5),
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

        self.stage1 = nn.Sequential(
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
        )
        self.down1 = WaveletDownsampler(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
        )

        self.stage2 = nn.Sequential(
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
        )
        self.down2 = WaveletDownsampler(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
        )

        self.stage3 = BothMamba(
            channels=hidden_dim,
            token_num=token_num,
            use_residual=use_residual,
            group_num=group_num,
            use_att=use_att,
        )

        self.pre_cls = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
        )
        self.cls_head = nn.Conv2d(
            in_channels=128,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.region_refiner = RegionConsistencyRefiner(
            guidance_channels=128,
            num_classes=num_classes,
            group_num=group_num,
            kernel_sizes=refine_kernel_sizes,
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)

        guidance = self.pre_cls(x)
        logits = self.cls_head(guidance)
        logits = self.region_refiner(guidance, logits)
        return logits
