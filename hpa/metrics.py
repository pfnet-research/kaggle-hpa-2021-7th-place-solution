import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import Tensor


def binary_focal_with_logits(y_pred: Tensor, y_true: Tensor, gamma: float = 2.0, reduction: str = "mean") -> Tensor:
    # NOTE: Assuming that the input is a hard label
    logpt = -F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
    pt = torch.exp(logpt)

    loss = (1 - pt) ** gamma * -logpt

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return torch.mean(loss)
    else:
        raise ValueError


def mAP(y_pred: Tensor, y_true: Tensor) -> float:
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return float(np.mean([average_precision_score(y_true[:, k], y_pred[:, k]) for k in range(y_pred.shape[1])]))
