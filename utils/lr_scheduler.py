import math
from typing import Any


class BaseLRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def _set_lrs(self, lrs: list[float]) -> list[float]:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = float(lr)
        return self.get_last_lr()

    def step(self, epoch: int) -> list[float]:
        raise NotImplementedError


class ConstantLRScheduler(BaseLRScheduler):
    def step(self, epoch: int) -> list[float]:
        del epoch
        return self._set_lrs(self.base_lrs)


class CosineLRScheduler(BaseLRScheduler):
    def __init__(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int = 0,
        min_lr: float | None = None,
        min_lr_ratio: float | None = None,
    ):
        super().__init__(optimizer)
        self.total_epochs = max(int(total_epochs), 1)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.min_lrs = self._resolve_min_lrs(min_lr=min_lr, min_lr_ratio=min_lr_ratio)

    def _resolve_min_lrs(
        self,
        *,
        min_lr: float | None,
        min_lr_ratio: float | None,
    ) -> list[float]:
        if min_lr is not None and min_lr_ratio is not None:
            raise ValueError("Specify either min_lr or min_lr_ratio, not both.")
        if min_lr is not None:
            min_lr = float(min_lr)
            if min_lr < 0:
                raise ValueError("min_lr must be >= 0.")
            return [min_lr for _ in self.base_lrs]
        if min_lr_ratio is not None:
            min_lr_ratio = float(min_lr_ratio)
            if not 0 <= min_lr_ratio <= 1:
                raise ValueError("min_lr_ratio must be in [0, 1].")
            return [base_lr * min_lr_ratio for base_lr in self.base_lrs]
        return [0.0 for _ in self.base_lrs]

    def step(self, epoch: int) -> list[float]:
        epoch = min(max(int(epoch), 0), self.total_epochs - 1)

        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            warmup_scale = float(epoch + 1) / float(self.warmup_epochs)
            return self._set_lrs([base_lr * warmup_scale for base_lr in self.base_lrs])

        cosine_total = max(self.total_epochs - self.warmup_epochs, 1)
        cosine_epoch = max(epoch - self.warmup_epochs, 0)
        cosine_progress = min(cosine_epoch / max(cosine_total - 1, 1), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

        lrs = [
            min_lr + (base_lr - min_lr) * cosine_factor
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]
        return self._set_lrs(lrs)


def build_lr_scheduler(
    optimizer,
    scheduler_cfg: Any,
    *,
    total_epochs: int,
):
    if scheduler_cfg is None:
        return ConstantLRScheduler(optimizer)

    name = str(
        getattr(
            scheduler_cfg,
            "name",
            getattr(scheduler_cfg, "type", "constant"),
        )
    ).lower()

    if name == "constant":
        return ConstantLRScheduler(optimizer)

    if name == "cosine":
        return CosineLRScheduler(
            optimizer,
            total_epochs=total_epochs,
            warmup_epochs=getattr(scheduler_cfg, "warmup_epochs", 0),
            min_lr=getattr(scheduler_cfg, "min_lr", None),
            min_lr_ratio=getattr(scheduler_cfg, "min_lr_ratio", None),
        )

    raise ValueError(
        f"Unsupported lr scheduler: {name}. Expected one of ['constant', 'cosine']."
    )
