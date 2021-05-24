
kaggle-hpa-2021-7th-place-solution
===

Code for 7th place solution of [Human Protein Atlas - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification) on Kaggle.

**A description of the method can be found in [this post](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238507) in the kaggle discussion.**


## Dataset Preparation

### Resize Images

```bash
# Resize train images to 768x768
python scripts/hap_segmenter/create_cell_mask.py resize_image \
    --input_directory data/input/hpa-single-cell-image-classification.zip/train \
    --output_directory data/input/hpa-768768.zip \
    --image_size 768
# Resize train images to 1536x1536
python scripts/hap_segmenter/create_cell_mask.py resize_image \
    --input_directory data/input/hpa-single-cell-image-classification.zip/train \
    --output_directory data/input/hpa-1536.zip \
    --image_size 1536

# Resize test images to 768x768
python scripts/hpa_segmenter/create_cell_mask.py resize_image \
    --input_directory /kaggle/input/hpa-single-cell-image-classification/test \
    --output_directory data/input/hpa-768-test.zip \
    --image_size 768
# Resize test images to 1536x1536
python scripts/hpa_segmenter/create_cell_mask.py resize_image \
    --input_directory /kaggle/input/hpa-single-cell-image-classification/test \
    --output_directory data/input/hpa-1536-test.zip \
    --image_size 1536
```

You can specify a directory in a zip file in the same way as a normal directory.

### Download Public HPA

Download all images in `kaggle_2021.tsv` in this [dataset](https://www.kaggle.com/lnhtrang/publichpa-withcellline), resize them into 768x768 and 1536x1536, and archive them as `data/input/hpa-public-768.zip` and `data/input/hpa-public-1536.zip`.

### Create Cell Mask

```bash
# Create cell masks for the Kaggle train set with 1536x1536
python scripts/hpa_segmenter/create_cell_mask.py create_cell_mask \
    --input_directory data/input/hpa-1536.zip \
    --output_directory data/input/hpa-1536-mask-v2.zip \
    --label_cell_scale_factor 1.0

# Resize the masks to 768x768
python scripts/hpa_segmenter/create_cell_mask.py resize_cell_mask \
    --input_directory data/input/hpa-1536-mask-v2.zip \
    --output_directory data/input/hpa-768-mask-v2-from-1536.zip \
    --image_size 768

# Create cell masks for the Public HPA dataset with 1536x1536
python scripts/hpa_segmenter/create_cell_mask.py create_cell_mask \
    --input_directory data/input/hpa-public-1536.zip/hpa-public-1536 \
    --output_directory data/input/hpa-public-1536-mask-v2.zip \
    --label_cell_scale_factor 1.0

# Resize the masks to 768x768
python scripts/hpa_segmenter/create_cell_mask.py resize_cell_mask \
    --input_directory data/input/hpa-public-1536-mask-v2.zip \
    --output_directory data/input/hpa-public-768-mask-v2-from-1536.zip \
    --image_size 768

# Create cell masks for the test set with the original resolution
# Run with `--label_cell_scale_factor = 0.5` to save inference time
python scripts/hpa_segmenter/create_cell_mask.py create_cell_mask \
    --input_directory /kaggle/input/hpa-single-cell-image-classification/test \
    --output_directory data/input/hpa-test-mask-v2.zip \
    --label_cell_scale_factor 0.5

# Resize the masks to 1536x1536
python scripts/hpa_segmenter/create_cell_mask.py resize_cell_mask \
    --input_directory data/input/hpa-test-mask-v2.zip \
    --output_directory data/input/hpa-test-mask-v2-1536.zip \
    --image_size 1536

# Resize the masks to 768x768
python scripts/hpa_segmenter/create_cell_mask.py resize_cell_mask \
    --input_directory data/input/hpa-test-mask-v2.zip \
    --output_directory data/input/hpa-test-mask-v2-768.zip \
    --image_size 768
```

### Create Input for Cell-level Classifier

```bash
# Create cell-level inputs for the Kaggle train set using 768x768 images as fixed scale image.
python scripts/hap_segmenter/create_cell_mask.py crop_and_resize_cell \
    --image_directory data/input/hpa-768768.zip \
    --cell_mask_directory data/input/hpa-768-mask-v2-from-1536.zip \
    --output_directory data/input/hpa-cell-crop-v2-192-from-768.zip \
    --image_size 192

# Create cell-level inputs for the Public HPA dataset using 768x768 images as fixed scale image.
python scripts/hap_segmenter/create_cell_mask.py crop_and_resize_cell \
    --image_directory data/input/hpa-public-768.zip \
    --cell_mask_directory data/input/hpa-public-768-mask-v2-from-1536.zip \
    --output_directory data/input/hpa-public-cell-crop-v2-192-from-768.zip \
    --image_size 192

# Create cell-level inputs for the Kaggle train set using 1536x1536 images as fixed scale image.
python scripts/hap_segmenter/create_cell_mask.py crop_and_resize_cell \
    --image_directory data/input/hpa-1536.zip \
    --cell_mask_directory data/input/hpa-1536-mask-v2.zip \
    --output_directory data/input/hpa-cell-crop-v2-192-from-1536.zip \
    --image_size 192

# Create cell-level inputs for the Public HPA dataset using 1536x1536 images as fixed scale image.
python scripts/hap_segmenter/create_cell_mask.py crop_and_resize_cell \
    --image_directory data/input/hpa-public-1536.zip \
    --cell_mask_directory data/input/hpa-public-1536-mask-v2.zip \
    --output_directory data/input/hpa-public-cell-crop-v2-192-from-1536.zip \
    --image_size 192

# Create cell-level inputs for the test set using 768x768 images as fixed scale image.
python scripts/hpa_segmenter/create_cell_mask.py crop_and_resize_cell \
    --image_directory data/input/hpa-768768-test.zip \
    --cell_mask_directory data/input/hpa-test-mask-v2-768.zip \
    --output_directory data/input/hpa-test-cell-crop-v2-192-from-768.zip \
    --image_size 192

# Create cell-level inputs for the test set using 1536x1536 images as fixed scale image.
python scripts/hpa_segmenter/create_cell_mask.py crop_and_resize_cell \
    --image_directory data/input/hpa-1536-test.zip \
    --cell_mask_directory data/input/hpa-test-mask-v2-1536.zip \
    --output_directory data/input/hpa-test-cell-crop-v2-192-from-1536.zip \
    --image_size 192
```

## Training

```bash
# Train image-level classifier
python scripts/cam_consistency_training/run.py train \
    --config_path scripts/cam_consistency_training/configs/${CONFIG_NAME}.yaml

# Train cell-level classifier
python scripts/cell_crop/run.py train \
    --config_path scripts/cell_crop/configs/${CONFIG_NAME}.yaml
```

If you want to train on multiple GPUs, use a launcher like [`torch.distributed.launch`](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py) and pass `--local_rank` option. You can override the fields in the config by passing an argument like `field_name=${value}` (e.g. `fold_index=1`). We trained 5 folds for all models used in the final submission pipeline. The config files are located in `scripts/cam_consistency_training/configs` and `scripts/cell_crop/configs`. We trained the models in the following order.

1. `scripts/cam_consistency_training/configs/eff-b2-focal-alpha1-cutmix-pubhpa-maskv2.yaml`
2. `scripts/cam_consistency_training/configs/eff-b5-focal-alpha1-cutmix-pubhpa-maskv2.yaml`
3. `scripts/cam_consistency_training/configs/eff-b7-focal-alpha1-cutmix-pubhpa-maskv2.yaml`
4. `scripts/cam_consistency_training/configs/eff-b2-cutmix-pubhpa-768-to-1536.yaml`
5. Do `predict_valid` and `concat_valid_predictions` (described below) for each model and save the average of the output files under `data/working/consistency_training/b2-1536-b2-b5-b7-768-avg/`.
7. `scripts/cam_consistency_training/configs/eff-b2-focal-stage2-b2b2b5b7avg.yaml`
6. `scripts/cell_crop/configs/resnest50-bce-from768-cutmix-softpl.yaml`
8. Do `predict_valid` and `concat_valid_predictions` for each model and save the average of the output files under `data/working/image-level-and-cell-crop-both-5folds/`.
9. `scripts/cam_consistency_training/configs/eff-b2-focal-stage3.yaml`
10. `scripts/cam_consistency_training/configs/eff-b2-focal-stage3-cos.yaml`
11. `scripts/cell_crop/configs/resnest50-bce-from768-stage3.yaml`
12. `scripts/cell_crop/configs/resnest50-bce-from1536-stage3-cos.yaml`

## Inference

### Validation Set

```bash
# Image-level classifier inference
python scripts/cam_consistency_training/run.py predict_valid \
    --config_path scripts/cam_consistency_training/configs/${CONFIG_NAME}.yaml

# Cell-level classifier inference
python scripts/cell_crop/run.py predict_valid \
    --config_path scripts/cell_crop/configs/${CONFIG_NAME}.yaml

# Concatenate the predictions for each fold to obtain the OOF prediction for the entire training data
python scripts/cam_consistency_training/run.py concat_valid_predictions \
    --config_path scripts/cam_consistency_training/configs/${CONFIG_NAME}.yaml
python scripts/cell_crop/run.py concat_valid_predictions \
    --config_path scripts/cell_crop/configs/${CONFIG_NAME}.yaml
```

### Test Set

```bash
# Image-level classifier inference
python scripts/cam_consistency_training/run.py predict_test \
    --config_path scripts/cam_consistency_training/configs/${CONFIG_NAME}.yaml

# Cell-level classifier inference
python scripts/cell_crop/run.py predict_test \
    --config_path scripts/cell_crop/configs/${CONFIG_NAME}.yaml

# Make our final submission with post-processing
python scripts/average_predictions.py \
    --orig_size_cell_mask_directory data/input/hpa-test-mask-v2.zip \
    "data/working/consistency_training/eff-b2-focal-stage3/0" \
    "data/working/consistency_training/eff-b2-focal-stage3/1" \
    "data/working/consistency_training/eff-b2-focal-stage3/2" \
    "data/working/consistency_training/eff-b2-focal-stage3/3" \
    "data/working/consistency_training/eff-b2-focal-stage3/4" \
    "data/working/consistency_training/eff-b2-focal-stage3-cos/0" \
    "data/working/consistency_training/eff-b2-focal-stage3-cos/1" \
    "data/working/consistency_training/eff-b2-focal-stage3-cos/2" \
    "data/working/consistency_training/eff-b2-focal-stage3-cos/3" \
    "data/working/consistency_training/eff-b2-focal-stage3-cos/4" \
    "data/working/cell_crop/resnest50-bce-from768-stage3/0" \
    "data/working/cell_crop/resnest50-bce-from768-stage3/1" \
    "data/working/cell_crop/resnest50-bce-from768-stage3/2" \
    "data/working/cell_crop/resnest50-bce-from768-stage3/3" \
    "data/working/cell_crop/resnest50-bce-from768-stage3/4" \
    "data/working/cell_crop/resnest50-bce-from1536-stage3-cos/0" \
    "data/working/cell_crop/resnest50-bce-from1536-stage3-cos/1" \
    "data/working/cell_crop/resnest50-bce-from1536-stage3-cos/2" \
    "data/working/cell_crop/resnest50-bce-from1536-stage3-cos/3" \
    "data/working/cell_crop/resnest50-bce-from1536-stage3-cos/4" \
    --edge_area_threshold 80000 --center_area_threshold 32000
```

## Use the code on Kaggle Notebook

Use docker to zip the source code and the wheels of the dependencies and upload them as a dataset.

```
docker run --rm -it -v /path/to/this/repo:/tmp/workspace -w /tmp/workspace/ gcr.io/kaggle-images/python bash ./build_zip.sh
```

In Kaggle Notebook, when you copy the code as shown below, you can run it the same way as your local environment.

```
# Make a working directory
!mkdir -p /kaggle/tmp

# Change the current directory
cd /kaggle/tmp

# Copy source code from the uploaded dataset
!cp -r /kaggle/input/<your-dataset-name>/* .

# You can use it as well as local environment
!python scripts/hpa_segmenter/create_cell_mask.py create_cell_mask ...
```
