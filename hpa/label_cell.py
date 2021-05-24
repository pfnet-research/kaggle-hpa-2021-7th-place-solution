"""
The original code for label_cell function is taken from HPA-Cell-Segmentation.
https://github.com/CellProfiling/HPA-Cell-Segmentation/blob/c7a2dcc30ba36453db8a4fb373dae90bc3844066/hpacellseg/utils.py#L83-L183
https://github.com/CellProfiling/HPA-Cell-Segmentation/blob/c7a2dcc30ba36453db8a4fb373dae90bc3844066/hpacellseg/utils.py#L29-L35

The code is distributed under Apache-2.0 License.
https://github.com/CellProfiling/HPA-Cell-Segmentation/blob/c7a2dcc30ba36453db8a4fb373dae90bc3844066/LICENSE

The code is modified by Kaizaburo Chubachi to process resized images.
"""

from typing import Tuple

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import filters, measure, segmentation
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects


def __fill_holes(image):
    """Fill_holes for labelled image, with a unique number."""
    boundaries = segmentation.find_boundaries(image)
    image = np.multiply(image, np.invert(boundaries))
    image = ndi.binary_fill_holes(image > 0)
    image = ndi.label(image)[0]
    return image


def _round_to_int(value: float) -> int:
    return int(np.round(value))


def label_cell_with_resized_pred(
    nuclei_pred: np.ndarray, cell_pred: np.ndarray, scale_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    area_scale_factor = scale_factor ** 2

    def __wsh(
        mask_img,
        threshold,
        border_img,
        seeds,
        threshold_adjustment=0.35,
        small_object_size_cutoff=10,
    ):
        img_copy = np.copy(mask_img)
        m = seeds * border_img  # * dt
        img_copy[m <= threshold + threshold_adjustment] = 0
        img_copy[m > threshold + threshold_adjustment] = 1
        img_copy = img_copy.astype(np.bool8)
        img_copy = remove_small_objects(img_copy, small_object_size_cutoff).astype(np.uint8)

        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        mask_img = mask_img.astype(np.bool8)
        mask_img = remove_small_holes(mask_img, _round_to_int(1000 * area_scale_factor))
        mask_img = remove_small_objects(mask_img, _round_to_int(8 * area_scale_factor)).astype(np.uint8)
        markers = ndi.label(img_copy, output=np.uint32)[0]
        labeled_array = segmentation.watershed(mask_img, markers, mask=mask_img, watershed_line=True)
        return labeled_array

    nuclei_label = __wsh(
        nuclei_pred[..., 2] / 255.0,
        0.4,
        1 - (nuclei_pred[..., 1] + cell_pred[..., 1]) / 255.0 > 0.05,
        nuclei_pred[..., 2] / 255,
        threshold_adjustment=-0.25,
        small_object_size_cutoff=_round_to_int(np.round(500 * area_scale_factor)),
    )

    # for hpa_image, to remove the small pseduo nuclei
    nuclei_label = remove_small_objects(nuclei_label, _round_to_int(2500 * area_scale_factor))
    nuclei_label = measure.label(nuclei_label)
    # this is to remove the cell borders' signal from cell mask.
    # could use np.logical_and with some revision, to replace this func.
    # Tuned for segmentation hpa images
    threshold_value = max(0.22, filters.threshold_otsu(cell_pred[..., 2] / 255) * 0.5)
    # exclude the green area first
    cell_region = np.multiply(
        cell_pred[..., 2] / 255 > threshold_value,
        np.invert(np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8)),
    )
    sk = np.asarray(cell_region, dtype=np.int8)
    distance = np.clip(cell_pred[..., 2], 255 * threshold_value, cell_pred[..., 2])
    cell_label = segmentation.watershed(-distance, nuclei_label, mask=sk)
    cell_label = remove_small_objects(cell_label, _round_to_int(5500 * area_scale_factor)).astype(np.uint8)
    selem = disk(_round_to_int(6 * scale_factor))
    cell_label = closing(cell_label, selem)
    cell_label = __fill_holes(cell_label)
    # this part is to use green channel, and extend cell label to green channel
    # benefit is to exclude cells clear on border but without nucleus
    sk = np.asarray(
        np.add(
            np.asarray(cell_label > 0, dtype=np.int8),
            np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8),
        )
        > 0,
        dtype=np.int8,
    )
    cell_label = segmentation.watershed(-distance, cell_label, mask=sk)
    cell_label = __fill_holes(cell_label)
    cell_label = np.asarray(cell_label > 0, dtype=np.uint8)
    cell_label = measure.label(cell_label)
    cell_label = remove_small_objects(cell_label, _round_to_int(5500 * area_scale_factor))
    cell_label = measure.label(cell_label)
    cell_label = np.asarray(cell_label, dtype=np.uint16)
    nuclei_label = np.multiply(cell_label > 0, nuclei_label) > 0
    nuclei_label = measure.label(nuclei_label)
    nuclei_label = remove_small_objects(nuclei_label, _round_to_int(2500 * area_scale_factor))
    nuclei_label = np.multiply(cell_label, nuclei_label > 0)

    return nuclei_label, cell_label


def label_cell_with_rescale(
    nuclei_pred: np.ndarray, cell_pred: np.ndarray, scale_factor: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    orig_size = nuclei_pred.shape[:2]
    assert cell_pred.shape[:2] == orig_size

    target_size = (_round_to_int(orig_size[0] * scale_factor), _round_to_int(orig_size[1] * scale_factor))
    nuclei_pred = cv2.resize(
        nuclei_pred,
        target_size,
        interpolation=cv2.INTER_AREA,
    )
    cell_pred = cv2.resize(
        cell_pred,
        target_size,
        interpolation=cv2.INTER_AREA,
    )

    nuclei_label, cell_label = label_cell_with_resized_pred(nuclei_pred, cell_pred, scale_factor)

    nuclei_label = cv2.resize(
        nuclei_label,
        orig_size,
        interpolation=cv2.INTER_AREA,
    )
    cell_label = cv2.resize(
        cell_label,
        orig_size,
        interpolation=cv2.INTER_AREA,
    )
    return nuclei_label, cell_label
