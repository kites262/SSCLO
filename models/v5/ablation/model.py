from torch import nn

from models.v5 import main as v5_main


class AvgDownsampler(nn.Module):
    """Baseline-style average-pooling downsampler used when the local enhancement module is disabled."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.proj(self.avg_pool(x))


class Model(v5_main.Model):
    """
    Module-level ablation wrapper around the stable v5 implementation.

    Two innovation modules are exposed as switches:
    1) local enhancement module:
       SpectralSelectionGate + AdaptiveLocalContextEnhancer + CenterAwareReweight + WaveletDownsamplerV2
    2) spatial consistency module:
       SpatialConsistencyRefiner
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
        use_local_enhancement_module: bool = True,
        use_spatial_consistency_module: bool = True,
        refiner_iters: int = 2,
    ):
        self.use_local_enhancement_module = bool(use_local_enhancement_module)
        self.use_spatial_consistency_module = bool(use_spatial_consistency_module)

        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_residual=use_residual,
            token_num=token_num,
            group_num=group_num,
            use_att=use_att,
            use_spectral_select=self.use_local_enhancement_module,
            use_center_gate=self.use_local_enhancement_module,
            use_refiner=self.use_spatial_consistency_module,
            refiner_iters=refiner_iters,
        )

        if not self.use_local_enhancement_module:
            self.stage1_lce = nn.Identity()
            self.stage2_lce = nn.Identity()
            self.stage3_lce = nn.Identity()
            self.down1 = AvgDownsampler(hidden_dim, hidden_dim)
            self.down2 = AvgDownsampler(hidden_dim, hidden_dim)


class LossCalculator(v5_main.LossCalculator):
    """Keeps the original v5 loss by default and disables TV/edge loss when the consistency module is ablated."""

    def __init__(
        self,
        label_smoothing: float = 0.1,
        tv_weight: float = 0.02,
        edge_weight: float = 0.05,
        pre_refine_key: str = "pre_refine_logits",
        edge_key: str = "edge",
        edge_prob_eps: float = 1e-4,
        ignore_index: int = -1,
        use_spatial_consistency_module: bool = True,
    ):
        self.use_spatial_consistency_module = bool(use_spatial_consistency_module)
        if not self.use_spatial_consistency_module:
            tv_weight = 0.0
            edge_weight = 0.0

        super().__init__(
            label_smoothing=label_smoothing,
            tv_weight=tv_weight,
            edge_weight=edge_weight,
            pre_refine_key=pre_refine_key,
            edge_key=edge_key,
            edge_prob_eps=edge_prob_eps,
            ignore_index=ignore_index,
        )
