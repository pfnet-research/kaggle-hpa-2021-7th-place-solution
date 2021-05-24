from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


def rand_bbox(size: Union[Tuple[int, int, int, int], torch.Size], lam: float) -> Tuple[int, int, int, int]:
    """Generate a random bounding box

    The original implemantation is below.
    https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L279-L295
    """

    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def apply_cutmix(
    x: Tensor, beta: float = 1.0, rand_index: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, int, int, int, int]:
    lam = np.random.beta(beta, beta)
    if rand_index is None:
        rand_index = torch.randperm(x.size()[0]).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    return x, rand_index, bbx1, bby1, bbx2, bby2


def inverse_cutmix(x: Tensor, rand_index: Tensor, bbx1: int, bby1: int, bbx2: int, bby2: int) -> Tensor:
    assert bbx1 < x.shape[2] and bbx2 <= x.shape[2]
    assert bby1 < x.shape[3] and bby2 <= x.shape[3]
    inv_rand_index = torch.argsort(rand_index)
    mixed_mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
    mixed_mask[:, :, bbx1:bbx2, bby1:bby2] = True
    return torch.where(mixed_mask, x[inv_rand_index], x)
