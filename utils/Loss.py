import warnings
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def head_loss(loss_func, logits, label, align_corners=False):
    seg_logits = resize(
        input=logits, size=label.shape[1:], mode="bilinear", align_corners=align_corners
    )

    loss = loss_func(seg_logits, label)
    return loss


def unpack_model_output(model_output: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if torch.is_tensor(model_output):
        return model_output, {}

    if isinstance(model_output, Mapping):
        if "logits" not in model_output:
            raise KeyError(
                "Model output dict must contain a 'logits' key for loss calculation."
            )
        aux = dict(model_output)
        logits = aux.pop("logits")
        if not torch.is_tensor(logits):
            raise TypeError("model_output['logits'] must be a torch.Tensor.")
        return logits, aux

    if isinstance(model_output, (tuple, list)):
        if len(model_output) != 2:
            raise TypeError(
                "Tuple/list model output must be (logits, aux_dict) or (aux_dict, logits)."
            )
        first, second = model_output
        if torch.is_tensor(first) and isinstance(second, Mapping):
            return first, dict(second)
        if isinstance(first, Mapping) and torch.is_tensor(second):
            aux = dict(first)
            aux.setdefault("logits", second)
            return unpack_model_output(aux)
        raise TypeError(
            "Unsupported tuple/list model output. Expected (Tensor, Mapping) or (Mapping, Tensor)."
        )

    raise TypeError(
        f"Unsupported model output type: {type(model_output).__name__}. Expected Tensor, Mapping, or tuple/list."
    )


def total_variation_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    dh = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
    dw = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
    return dh + dw


def edge_target_from_mask(
    mask: torch.Tensor,
    ignore_index: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.dim() == 4:
        mask = mask[:, 0]
    mask = mask.long()

    valid = mask != ignore_index
    safe_mask = mask.clone()
    safe_mask[~valid] = 0

    dh_valid = valid[:, 1:, :] & valid[:, :-1, :]
    dw_valid = valid[:, :, 1:] & valid[:, :, :-1]
    dh = ((safe_mask[:, 1:, :] != safe_mask[:, :-1, :]) & dh_valid).float()
    dw = ((safe_mask[:, :, 1:] != safe_mask[:, :, :-1]) & dw_valid).float()

    edge = torch.zeros(
        (mask.shape[0], mask.shape[1], mask.shape[2]),
        device=mask.device,
        dtype=torch.float32,
    )
    edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], dh)
    edge[:, :-1, :] = torch.maximum(edge[:, :-1, :], dh)
    edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], dw)
    edge[:, :, :-1] = torch.maximum(edge[:, :, :-1], dw)

    valid_mask = valid.unsqueeze(1).float()
    return edge.unsqueeze(1) * valid_mask, valid_mask


def merge_loss_logs(*log_dicts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for log_dict in log_dicts:
        for key, value in log_dict.items():
            if key not in merged:
                merged[key] = value
            else:
                merged[key] = merged[key] + value
    return merged


def loss_logs_to_scalars(
    log_dict: dict[str, Any],
    prefix: str = "train/",
) -> dict[str, float]:
    scalar_logs: dict[str, float] = {}
    for key, value in log_dict.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            scalar_logs[f"{prefix}{key}"] = float(value.detach().cpu())
        elif isinstance(value, (float, int)):
            scalar_logs[f"{prefix}{key}"] = float(value)
    return scalar_logs


class CrossEntropyLossCalculator:
    def run_model(self, net, x: torch.Tensor):
        return net(x)

    def __call__(
        self,
        *,
        loss_func,
        model_output,
        label: torch.Tensor,
    ) -> dict[str, Any]:
        logits, aux = unpack_model_output(model_output)
        ce_loss = head_loss(loss_func, logits, label)
        return {
            "loss": ce_loss,
            "logits": logits,
            "aux": aux,
            "log_items": {
                "loss": ce_loss,
                "ce_loss": ce_loss,
            },
        }

