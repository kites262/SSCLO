from torch import nn

from models.MambaHSI import BothMamba
from models.v1.fno import FourierLayer2D


class MambaFNO_Head(nn.Module):
    def __init__(self, in_channels, embedding_channels=64, num_classes=10):
        super(MambaFNO_Head, self).__init__()

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
