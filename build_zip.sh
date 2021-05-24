rm -rf dependencies
rm -f source.zip

pip wheel -w dependencies --no-deps -r requirements_for_notebook.txt
pip wheel -w dependencies --no-deps "git+https://github.com/haoxusci/pytorch_zoo@master#egg=pytorch_zoo"

pip wheel -w dependencies --no-deps ./HPA-Cell-Segmentation
pip wheel -w dependencies --no-deps ./somen

zip -r source.zip \
  hpa \
  dependencies \
  scripts/hpa_segmenter/create_cell_mask.py \
  scripts/average_predictions.py \
  scripts/cam_consistency_training/run.py \
  scripts/cam_consistency_training/configs/eff-b2-focal-stage3.yaml \
  scripts/cam_consistency_training/configs/eff-b2-focal-stage3-cos.yaml \
  scripts/cell_crop/run.py \
  scripts/cell_crop/configs/resnest50-bce-from768-stage3.yaml \
  scripts/cell_crop/configs/resnest50-bce-from1536-stage3-cos.yaml \
  -x \*/__pycache__/\* \
  -x \*/.\* \
  -x \*/.\*/\* \
  -x .\*
