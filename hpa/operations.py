from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def cell_wise_average_pooling(x: Tensor, cell_mask: Tensor) -> Tuple[Tensor, Tensor]:
    B, C, H, W = x.shape
    # N = cell_mask.max().item()
    N = 255
    Np1 = N + 1  # There are N + 1 types of cell_mask values (0, ..., N)
    device = x.device

    cell_mask = cell_mask.to(torch.int64).reshape(B, 1, H, W)

    indices = (
        C * Np1 * torch.arange(B, device=device)[:, np.newaxis, np.newaxis, np.newaxis]
        + Np1 * torch.arange(C, device=device)[np.newaxis, :, np.newaxis, np.newaxis]
        + cell_mask
    )
    assert 0 <= indices.max().item() < B * C * Np1

    x_sum = torch.zeros((B * C * Np1), device=device)
    x_sum.scatter_add_(0, indices.view(-1), x.view(-1))

    pixel_count = torch.zeros((B * C * Np1), device=device)
    pixel_count.scatter_add_(0, indices.view(-1), torch.ones_like(x.view(-1)))

    # Exclude 0 because it is not an instance number in cell_mask
    x_sum = x_sum.reshape(B, C, Np1)[..., 1:]
    pixel_count = pixel_count.reshape(B, C, Np1)[..., 1:]

    x_mean = x_sum / (pixel_count + 1e-6)
    return x_mean, pixel_count


def _test_cell_wise_average_pooling() -> None:
    B, C, H, W = 3, 5, 32, 32
    x = torch.rand((B, C, H, W))

    g = torch.Generator()
    g.manual_seed(0)

    cell_mask = torch.randint(0, 10, (B, H, W), generator=g)
    # max_instance_index = cell_mask.max().item()
    max_instance_index = 255

    actual_y, actual_pixel_count = cell_wise_average_pooling(x, cell_mask)

    expected_y = torch.zeros((B, C, max_instance_index))
    expected_pixel_count = torch.zeros((B, C, max_instance_index))
    for instance_index in range(max_instance_index):
        bool_mask = cell_mask == (instance_index + 1)
        expected_pixel_count[:, :, instance_index] = torch.sum(bool_mask[:, np.newaxis, :, :], dim=(2, 3))
        expected_y[:, :, instance_index] = torch.sum(x * bool_mask[:, np.newaxis, :, :], dim=(2, 3)) / (
            expected_pixel_count[:, :, instance_index] + 1e-6
        )

    assert torch.allclose(expected_y, actual_y)
    assert torch.allclose(expected_pixel_count, actual_pixel_count)
    print("Pass test_cell_wise_average_pooling")


def calc_conf_scale_on_single_image(cell_mask: Tensor, edge_area_threshold: int, center_area_threshold: int) -> Tensor:
    """Calculate the `conf_scale` used in postprocess for the cell_mask: H x W of a single image"""

    N = 255
    size = int(cell_mask.max())
    device = cell_mask.device

    conf_scales = torch.ones(size, device=device)
    conf_scales = F.pad(conf_scales, [0, N - len(conf_scales)], "constant", 0)  # Align length to 255

    for cell_idx in range(1, size + 1):  # 1 <= cell index <= size
        binary_mask = cell_mask == cell_idx
        binary_mask = binary_mask.to(device)
        is_edge = torch.any(torch.cat([binary_mask[0], binary_mask[-1], binary_mask[:, 0], binary_mask[:, -1]]))

        if is_edge:
            if edge_area_threshold > 0:
                # Scale confidence for small cells on the edge
                area = torch.sum(binary_mask)
                confidence_scale: Union[float, Tensor] = min(1.0, area / edge_area_threshold)  # type: ignore

                if confidence_scale < 1.0:
                    conf_scales[cell_idx - 1] = confidence_scale
        else:
            # It is center
            if center_area_threshold > 0:
                # Scale confidence for small cells on the edge
                area = torch.sum(binary_mask)
                confidence_scale = min(1.0, area / center_area_threshold)  # type: ignore

                if confidence_scale < 1.0:
                    conf_scales[cell_idx - 1] = confidence_scale

    return conf_scales


def calc_conf_scale_on_batch(cell_mask: Tensor, edge_area_threshold: int, center_area_threshold: int) -> Tensor:
    """
    Calculate `conf_scale` for cell_mask: B x H x W
    >> best public LB (0.573) threshold -> --edge_area_threshold 80000 --center_area_threshold 32000
    >> Since more than 80% of the pixels are 2048,
       set (768 / 2048) ** 2 == 0.140625 as the reduction rate of the edge area threshold.
    >> 768 x 768 -> --edge_area_threshold 11250 --center_area_threshold 4500
    """

    conf_scales_list: List[Tensor] = []
    label_num = 19
    bs = len(cell_mask)

    for cm_single in cell_mask:
        conf_scales = calc_conf_scale_on_single_image(cm_single, edge_area_threshold, center_area_threshold)
        conf_scales_list.append(conf_scales)

    conf_scales_batch = torch.stack(conf_scales_list).reshape(bs, 1, -1)  # bs x 255 -> bs x 1 x 255
    conf_scales_batch = conf_scales_batch.repeat(1, label_num, 1).detach()  # -> bs x 19 x 255

    return conf_scales_batch


def _test_calc_conf_scale_on_single_image():
    edge_area_threshold = 10
    center_area_threshold = 5
    device = "cpu"

    cell_mask = torch.zeros(10, 10).to(device=device)

    # on edge
    cell_mask[:2, :7] = 1  # big (2x7)
    cell_mask[7:9, 9:] = 2  # small (2x2)

    # on center
    cell_mask[5:8, 5:8] = 3  # big (3x3)
    cell_mask[3, 4:6] = 4  # small (1x2)

    conf_scales = calc_conf_scale_on_single_image(cell_mask, edge_area_threshold, center_area_threshold)

    assert conf_scales[0] == 1
    assert conf_scales[1] == 0.2
    assert conf_scales[2] == 1
    assert conf_scales[3] == 0.4

    print("ok: _test_calc_conf_scale_on_single_image")


def _test_calc_conf_scale_on_batch():
    edge_area_threshold = 10
    center_area_threshold = 5
    device = "cpu"

    cell_mask = torch.zeros(2, 10, 10).to(device=device)

    # sample 0
    # on edge
    cell_mask[0, :2, :7] = 1  # big (2x7)
    cell_mask[0, 7:9, 9:] = 2  # small (2x2)

    # on center
    cell_mask[0, 5:8, 5:8] = 3  # big (3x3)
    cell_mask[0, 3, 4:6] = 4  # small (1x2)

    # sample 1
    # on edge
    cell_mask[1, :3, :8] = 1  # big (3x8)
    cell_mask[1, 9, 7:] = 2  # small (1x3)

    # on center
    cell_mask[1, 5:8, 5:8] = 3  # big (3x3)
    cell_mask[1, 2, 5:6] = 4  # small (1x1)

    conf_scales = calc_conf_scale_on_batch(cell_mask, edge_area_threshold, center_area_threshold)

    assert torch.equal(conf_scales[0, 0, :5], torch.tensor([1.0, 0.2, 1.0, 0.4, 0]))
    assert torch.equal(conf_scales[1, 0, :5], torch.tensor([1.0, 0.3, 1.0, 0.2, 0]))
    print("ok: _test_calc_conf_scale_on_batch")


if __name__ == "__main__":
    _test_cell_wise_average_pooling()
    _test_calc_conf_scale_on_single_image()
    _test_calc_conf_scale_on_batch()
