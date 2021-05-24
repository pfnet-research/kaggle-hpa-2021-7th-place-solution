import os
from pathlib import Path
from typing import Sequence

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

import somen
from hpa.encode_binary_mask import encode_binary_mask
from hpa.reading import read_gray
from somen.pfio_utility import DirectoryInZip


def average_predictions(
    orig_size_cell_mask_directory: str,
    *working_dirs: Sequence[str],
    edge_area_threshold: int = -1,
    center_area_threshold: int = -1,
    out_filepath: str = "submission.csv",
) -> None:
    print("working_dirs", working_dirs)
    print("edge_area_threshold", edge_area_threshold)

    concat_predictions, concat_instance_indices, sizes = None, None, None
    for working_dir in map(Path, working_dirs):
        concat_predictions_child = somen.file_io.load_array(working_dir / "predictions_test.h5")
        concat_instance_indices_child = somen.file_io.load_array(working_dir / "instance_indices_test.h5")
        sizes_child = somen.file_io.load_array(working_dir / "sizes_test.h5")

        concat_predictions_child = concat_predictions_child.astype(np.float64)
        if concat_predictions_child.ndim == 3:
            concat_predictions_child = concat_predictions_child.mean(axis=-1)

        if concat_predictions is None:
            assert concat_instance_indices is None
            assert sizes is None

            concat_predictions = concat_predictions_child / len(working_dirs)
            concat_instance_indices = concat_instance_indices_child
            sizes = sizes_child
        else:
            assert concat_instance_indices is not None
            assert sizes is not None

            concat_predictions += concat_predictions_child / len(working_dirs)
            assert (concat_instance_indices == concat_instance_indices_child).all()
            assert (sizes == sizes_child).all()

    orig_size_cell_mask_directory = DirectoryInZip(orig_size_cell_mask_directory)
    image_ids = np.unique([filename.split(".")[0] for filename in orig_size_cell_mask_directory.listdir()])
    image_ids = sorted(image_ids)

    assert len(image_ids) == len(sizes)

    edge_count = 0
    center_count = 0
    edge_scaled_count = 0
    center_scaled_count = 0

    submission = []
    j = 0
    for i, image_id in enumerate(tqdm(image_ids)):
        cell_mask = read_gray(orig_size_cell_mask_directory, f"{image_id}.png")

        row = {}
        row["ID"] = image_id
        row["ImageWidth"] = cell_mask.shape[1]
        row["ImageHeight"] = cell_mask.shape[0]

        pred_strs = []
        for _ in range(sizes[i]):
            binary_mask = cell_mask == concat_instance_indices[j]
            is_edge = np.any(np.concatenate([binary_mask[0], binary_mask[-1], binary_mask[:, 0], binary_mask[:, -1]]))
            if is_edge:
                if edge_area_threshold > 0:
                    # Scale confidence for small cells on the edge
                    area = np.sum(binary_mask)
                    confidence_scale = min(1.0, area / edge_area_threshold)
                    if confidence_scale < 1.0:
                        print(
                            f"area {area}, edge_area_threshold {edge_area_threshold}, confidence_scale {confidence_scale}"
                        )
                        edge_scaled_count += 1
                    concat_predictions[j] *= confidence_scale
                edge_count += 1
            else:
                # It is center
                if center_area_threshold > 0:
                    # Scale confidence for small cells on the edge
                    area = np.sum(binary_mask)
                    confidence_scale = min(1.0, area / center_area_threshold)
                    if confidence_scale < 1.0:
                        print(
                            f"area {area}, center_area_threshold {center_area_threshold}, confidence_scale {confidence_scale}"
                        )
                        center_scaled_count += 1
                    concat_predictions[j] *= confidence_scale
                center_count += 1

            mask_str = encode_binary_mask(binary_mask)
            for k in range(19):
                pred_strs.extend([str(k), str(concat_predictions[j, k]), mask_str])
            j += 1
        row["PredictionString"] = " ".join(pred_strs)

        submission.append(row)

    dirpath = os.path.dirname(out_filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    pd.DataFrame(submission).to_csv(out_filepath, index=False)
    print(
        "center_count",
        center_count,
        "edge_count",
        edge_count,
        "center_scaled_count",
        center_scaled_count,
        "edge_scaled_count",
        edge_scaled_count,
    )
    print(f"Done. Saved to {out_filepath}")


if __name__ == "__main__":
    fire.Fire(average_predictions)
