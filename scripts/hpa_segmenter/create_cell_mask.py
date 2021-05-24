import pickle
import tempfile
import time
from multiprocessing import Process, Queue
from queue import Empty
from typing import Sequence

import blosc
import cv2
import fire
import hpacellseg.cellsegmentator as cellsegmentator
import numpy as np
import skimage
import torch
import torch.nn.functional as F
from hpacellseg.cellsegmentator import NORMALIZE
from tqdm import tqdm

from hpa.label_cell import label_cell_with_resized_pred
from hpa.reading import read_cellseg_input, read_gray
from somen.pfio_utility import DirectoryInZip, setup_forkserver


def _cv2_imwrite(directory: DirectoryInZip, path: str, image: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".png") as tmp_fp:
        cv2.imwrite(tmp_fp.name, image)
        with open(tmp_fp.name, "rb") as read_fp, directory.open(path, mode="wb") as write_fp:
            write_fp.write(read_fp.read())


def _segmentation_worker(
    in_queue: Queue, out_queue: Queue, input_directory: str, nuc_model: str, cell_model: str, batch_size: int
) -> None:
    input_directory = DirectoryInZip(input_directory)
    device = "cuda"

    segmentator = cellsegmentator.CellSegmentator(
        nuc_model,
        cell_model,
        scale_factor=0.25,
        device=device,
        padding=False,
        multi_channel_model=True,
    )

    MEAN = torch.tensor(NORMALIZE["mean"], device=device)[np.newaxis, :, np.newaxis, np.newaxis]
    STD = torch.tensor(NORMALIZE["std"], device=device)[np.newaxis, :, np.newaxis, np.newaxis]

    done = False

    while not done:

        image_ids = []
        while len(image_ids) < batch_size:
            image_id = in_queue.get()
            if image_id is None:
                done = True
                break
            image_ids.append(image_id)

        if len(image_ids) == 0:
            continue

        uint_images = [read_cellseg_input(input_directory, image_id) for image_id in image_ids]
        shapes = [image.shape[:2] for image in uint_images]
        images = torch.tensor(
            [np.moveaxis(skimage.transform.resize(image, (512, 512)), 2, 0) for image in uint_images],
            device=device,
            dtype=torch.float32,
        )
        images = (images - MEAN) / STD

        with torch.no_grad():
            nuc_seg = F.softmax(segmentator.nuclei_model(images[:, [2, 2, 2]]), dim=1)
            nuc_seg[:, 0] = 0
            nuc_seg = nuc_seg.detach().cpu().numpy()
            nuc_seg = np.rint((nuc_seg * 255)).clip(0, 255).astype(np.uint8)
            nuc_seg = np.moveaxis(nuc_seg, 1, 3)

            cell_seg = F.softmax(segmentator.cell_model(images), dim=1)
            cell_seg[:, 0] = 0
            cell_seg = cell_seg.detach().cpu().numpy()
            # For some unknown reason, restore_scaling_padding -> img_as_ubyte is applied to cell_seg,
            # so we don't convert it to int here
            cell_seg = np.moveaxis(cell_seg, 1, 3)

        for i, image_id in enumerate(image_ids):
            out_queue.put((image_id, cell_seg[i], nuc_seg[i], shapes[i]))


def _postprocess_worker(in_queue: Queue, out_queue: Queue, label_cell_scale_factor: float) -> None:
    while True:
        msg = in_queue.get()
        if msg is None:
            break

        image_id, cell_seg, nuc_seg, orig_shape = msg

        # If the resolution is too small, the result will be quite different due to integer round errors
        # of `disk` in the label_cell, so make it larger than that when putting it into the segmentator.
        image_size = int(2048 * label_cell_scale_factor)

        cell_seg = cv2.resize(cell_seg, (image_size, image_size), interpolation=cv2.INTER_AREA)
        cell_seg = np.rint((cell_seg * 255)).clip(0, 255).astype(np.uint8)
        nuc_seg = cv2.resize(nuc_seg, (image_size, image_size), interpolation=cv2.INTER_AREA)

        _, cell_mask = label_cell_with_resized_pred(nuc_seg, cell_seg, label_cell_scale_factor)

        assert 0 <= cell_mask.min() and cell_mask.max() <= 255
        cell_mask = cell_mask.astype(np.uint8)
        cell_mask = cv2.resize(cell_mask, orig_shape, interpolation=cv2.INTER_NEAREST_EXACT)

        out_queue.put((image_id, cell_mask))


def create_cell_mask(
    input_directory: str,
    output_directory: str,
    batch_size: int = 16,
    num_segmentation_workers: int = 1,
    num_postprocess_workers: int = 16,
    label_cell_scale_factor: float = 1.0,
    nuc_model: str = "../nuclei-model.pth",
    cell_model: str = "../cell-model.pth",
) -> None:
    input_directory = DirectoryInZip(input_directory)
    output_directory = DirectoryInZip(output_directory)

    image_id_queue, seg_queue, mask_queue = Queue(), Queue(maxsize=batch_size), Queue()

    image_ids = np.unique(["_".join(filename.split("_")[:-1]) for filename in input_directory.listdir()])
    image_ids = sorted(image_ids)

    for image_id in image_ids:
        image_id_queue.put(image_id)

    segmentation_workers = []
    for _ in range(num_segmentation_workers):
        p = Process(
            target=_segmentation_worker,
            args=(image_id_queue, seg_queue, str(input_directory), nuc_model, cell_model, batch_size),
        )
        p.start()
        segmentation_workers.append(p)

        image_id_queue.put(None)

    postprocess_workers = []
    for _ in range(num_postprocess_workers):
        p = Process(target=_postprocess_worker, args=(seg_queue, mask_queue, label_cell_scale_factor))
        p.start()
        postprocess_workers.append(p)

    done_segmentation, done_postprocess = False, False

    pbar = tqdm(total=len(image_ids))

    while True:
        try:
            image_id, cell_mask = mask_queue.get_nowait()
            _cv2_imwrite(output_directory, f"{image_id}.png", cell_mask)
            pbar.update(1)
            continue
        except Empty:
            pass

        time.sleep(5)

        if not done_segmentation:
            if all([not p.is_alive() for p in segmentation_workers]):
                done_segmentation = True
                for _ in postprocess_workers:
                    seg_queue.put(None)
        elif not done_postprocess:
            if all([not p.is_alive() for p in postprocess_workers]):
                done_postprocess = True
        else:
            assert done_segmentation and done_postprocess
            break

    pbar.close()


def resize_cell_mask(input_directory: str, output_directory: str, image_size: int) -> None:
    input_directory = DirectoryInZip(input_directory)
    output_directory = DirectoryInZip(output_directory)

    for path in tqdm(sorted(input_directory.listdir())):
        cell_mask = read_gray(input_directory, path)
        cell_mask = cv2.resize(cell_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST_EXACT)
        _cv2_imwrite(output_directory, path, cell_mask)


def resize_image(input_directory: str, output_directory: str, image_size: int) -> None:
    input_directory = DirectoryInZip(input_directory)
    output_directory = DirectoryInZip(output_directory)

    for path in tqdm(sorted(input_directory.listdir())):
        img = read_gray(input_directory, path)
        img = cv2.resize(img, (image_size, image_size))
        _cv2_imwrite(output_directory, path, img)


def _crop_and_resize_cell_worker(
    image_directory: str,
    cell_mask_directory: str,
    image_size: int,
    in_queue: Queue,
    out_queue: Queue,
) -> Sequence[int]:
    image_directory = DirectoryInZip(image_directory)
    cell_mask_directory = DirectoryInZip(cell_mask_directory)

    while True:
        msg = in_queue.get()
        if msg is None:
            break

        image_id = msg

        cell_mask = read_gray(cell_mask_directory, f"{image_id}.png")
        assert cell_mask.ndim == 2

        if (cell_mask == 0).all():
            out_queue.put((image_id, None, []))
            continue

        cell_images = []
        instance_indices = []

        for instance_index in range(1, cell_mask.max() + 1):
            instance_mask = cell_mask == instance_index
            ys, xs = np.where(instance_mask)
            if len(ys) == 0:
                continue

            instance_indices.append(instance_index)

            whole_slice_y = slice(ys.min(), ys.max() + 1)
            whole_slice_x = slice(xs.min(), xs.max() + 1)

            images = {}
            for color in ["red", "green", "blue", "yellow"]:
                image = read_gray(image_directory, f"{image_id}_{color}.png")
                assert image.shape == cell_mask.shape
                images[color] = image

            weight = images["blue"][ys, xs]
            weight = weight / (weight.sum() + 1e-6)
            center_y = int((weight * ys).sum())
            center_x = int((weight * xs).sum())

            # Crop around nuclei without resizing (not necessarily the whole cell)
            def _get_nuclei_center_crop(src: np.ndarray) -> np.ndarray:
                dst_y_start = 0
                src_y_start = center_y - image_size // 2 + 1
                if src_y_start < 0:
                    dst_y_start = -src_y_start
                    src_y_start = 0
                dst_x_start = 0
                src_x_start = center_x - image_size // 2 + 1
                if src_x_start < 0:
                    dst_x_start = -src_x_start
                    src_x_start = 0

                dst_y_end = image_size
                src_y_end = center_y + image_size // 2 + 1
                if src_y_end >= cell_mask.shape[0]:
                    dst_y_end = image_size - (src_y_end - cell_mask.shape[0])
                    src_y_end = cell_mask.shape[0]
                dst_x_end = image_size
                src_x_end = center_x + image_size // 2 + 1
                if src_x_end >= cell_mask.shape[1]:
                    dst_x_end = image_size - (src_x_end - cell_mask.shape[1])
                    src_x_end = cell_mask.shape[1]

                dst = np.zeros((image_size, image_size), dtype=src.dtype)
                dst[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = src[src_y_start:src_y_end, src_x_start:src_x_end]
                return dst

            # Crop whole cell with resizing
            def _get_resized_whole_crop(src: np.ndarray) -> np.ndarray:
                whole_crop = src[whole_slice_y, whole_slice_x]

                h, w = whole_crop.shape
                ratio = image_size / max(h, w)
                h_new, w_new = int(h * ratio), int(w * ratio)
                assert h_new <= image_size and w_new <= image_size

                resized = np.zeros((image_size, image_size), dtype=image.dtype)
                y_start = (image_size - h_new) // 2
                x_start = (image_size - w_new) // 2
                # NOTE: cv2.resize uses it in (x, y) order, so it becomes (w_new, h_new)
                resized[y_start : y_start + h_new, x_start : x_start + w_new] = cv2.resize(whole_crop, (w_new, h_new))
                return resized

            for color in ["red", "green", "blue", "yellow"]:
                image = images[color]
                image = image * instance_mask

                nuclei_center_crop = _get_nuclei_center_crop(image)
                resized_whole_crop = _get_resized_whole_crop(image)

                cell_images.append(nuclei_center_crop)
                cell_images.append(resized_whole_crop)

        cell_images = np.asarray(cell_images).reshape(-1, 4 * 2, image_size, image_size)

        # Pack to save time in serialize (not sure how effective it is)
        out_queue.put((image_id, blosc.pack_array(cell_images), instance_indices))


def crop_and_resize_cell(
    image_directory: str, cell_mask_directory: str, output_directory: str, image_size: int, num_workers: int = 2
) -> None:
    image_directory = DirectoryInZip(image_directory)
    cell_mask_directory = DirectoryInZip(cell_mask_directory)
    output_directory = DirectoryInZip(output_directory)

    image_ids = np.unique([filename[: filename.rindex("_")] for filename in image_directory.listdir()])
    image_ids = sorted(image_ids)

    in_queue, out_queue = Queue(), Queue(maxsize=128)

    for image_id in image_ids:
        in_queue.put(image_id)
    for _ in range(num_workers):
        in_queue.put(None)

    workers = []
    for _ in range(num_workers):
        p = Process(
            target=_crop_and_resize_cell_worker,
            args=(str(image_directory), str(cell_mask_directory), image_size, in_queue, out_queue),
        )
        p.start()
        workers.append(p)

    done = False
    instance_indices_by_id = {}
    pbar = tqdm(total=len(image_ids))

    while True:
        try:
            image_id, cell_images, instance_indices = out_queue.get_nowait()

            if len(instance_indices) > 0:
                assert cell_images is not None

                cell_images = blosc.unpack_array(cell_images)
                assert len(instance_indices) == cell_images.shape[0]

                for i, instance_index in enumerate(instance_indices):
                    with output_directory.open(f"{image_id}_{instance_index}.blosc", mode="wb") as fp:
                        fp.write(blosc.pack_array(cell_images[i]))

            instance_indices_by_id[image_id] = instance_indices
            pbar.update(1)
            continue
        except Empty:
            pass

        time.sleep(1)
        if not done:
            if all([not p.is_alive() for p in workers]):
                done = True
                # Don't break, run the while loop again until get_nowait fails
        else:
            break

    pbar.close()
    with output_directory.open("instance_indices_by_id.pkl", "wb") as fp:
        pickle.dump(instance_indices_by_id, fp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    setup_forkserver()
    fire.Fire(
        {
            "create_cell_mask": create_cell_mask,
            "resize_cell_mask": resize_cell_mask,
            "resize_image": resize_image,
            "crop_and_resize_cell": crop_and_resize_cell,
        }
    )
