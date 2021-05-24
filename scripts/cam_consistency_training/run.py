import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import fire
import numpy as np
import pandas as pd
import PIL
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn.functional as F
import torchvision
from ignite.engine import Engine
from ignite.utils import convert_tensor
from rich import print
from torch import Tensor, nn
from tqdm import tqdm

import somen
import somen.albumentations_utility as As
from hpa.augmentation import create_green_channel_albumentations
from hpa.cutmix import apply_cutmix, inverse_cutmix
from hpa.metrics import binary_focal_with_logits, mAP
from hpa.operations import calc_conf_scale_on_batch, cell_wise_average_pooling
from hpa.reading import read_gray, read_rgby
from hpa.validation import get_folds
from somen.pfio_utility import DirectoryInZip, setup_forkserver
from somen.pytorch_utility.extensions.wandb import WandbReporter, wandb_init


@dataclass
class ModelConfig:
    arch: str
    instance_norm_input: bool = False
    remove_maxpool: bool = False
    remove_first_stride2: bool = False
    pretrained: Optional[str] = None
    use_cosine_classifier: bool = False


@dataclass
class OptimizationConfig:
    objective: str
    optimizer: str
    optimizer_params: Mapping[str, Any]
    batch_size: int
    nb_epoch: int
    lr_scheduler: Optional[str]
    lr_scheduler_params: Optional[Mapping[str, Any]]
    sampler_seed: int = 0


@dataclass
class ConsistencyTrainingCoonfig:
    drop_cell_p: float
    cutmix_p: float
    cutmix_p_in_heavier_aug: float
    cutmix_repeat: int = 1
    shear_range_abs: float = 16.0
    angle_limit: float = 45.0
    scale_limit: float = 0.1
    max_alpha: Optional[float] = 1.0
    alpha_schedule: Optional[float] = 0.0
    consistency_loss_criteria: str = "CE"
    consistency_loss_targets: str = "all"
    pixel_wise_heavier_aug: Optional[Mapping[str, Any]] = None
    hard_label_threshold: Optional[float] = None
    weight_image_level_class_loss: float = 1.0
    p_er: float = 0.0  # ER channel augmentation
    p_mt: float = 0.0  # Microtubule channel augmentation
    ermt_transform: Optional[Mapping[str, Any]] = None  # albumentations augmentation for ER/MT channel creation
    edge_area_threshold: int = -1
    center_area_threshold: int = -1


@dataclass
class DatasetConfig:
    image_directory: str
    cell_mask_directory: str
    orig_size_cell_mask_directory: Optional[str]
    pseudo_cell_level_label_dir: Optional[Path]


@dataclass
class TrainingConfig:
    n_folds: int
    fold_index: int
    model: ModelConfig
    optim: OptimizationConfig
    train_dataset: DatasetConfig
    public_hpa_dataset: Optional[DatasetConfig]
    test_dataset: DatasetConfig
    train_transform: Mapping[str, Any]
    valid_transform: Mapping[str, Any]
    test_transform: Mapping[str, Any]
    consistency_training: ConsistencyTrainingCoonfig


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_ids: Sequence[str],
        labels: Sequence[str],
        image_directory: str,
        cell_mask_directory: str,
        transform: A.Compose,
        # Augmentation that shifts the positional relationship is performed in update_fn,
        # so only pixel-wise augmentation is performed in the Dataset.
        pixel_wise_heavier_aug: Optional[A.Compose],
        drop_cell_p: float,
        pseudo_cell_level_labels: Optional[Sequence[np.ndarray]] = None,
        p_er: float = 0.0,
        p_mt: float = 0.0,
        ermt_transform: Optional[A.Compose] = None,
    ) -> None:
        self.image_ids = image_ids
        self.labels = labels
        self.image_directory = DirectoryInZip(image_directory)
        self.cell_mask_directory = DirectoryInZip(cell_mask_directory)
        self.transform = transform
        self.pixel_wise_heavier_aug = pixel_wise_heavier_aug
        self.drop_cell_p = drop_cell_p
        self.pseudo_cell_level_labels = pseudo_cell_level_labels
        self.p_er = p_er
        self.p_mt = p_mt
        self.ermt_transform = ermt_transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        label = self.labels[index]

        rgby = read_rgby(self.image_directory, image_id, channel_last=True).astype(np.float32)
        cell_mask = read_gray(self.cell_mask_directory, f"{image_id}.png")

        image_level_label = np.zeros(19, dtype=np.float32)

        tmp = np.random.uniform()

        if tmp < self.p_er:
            # Replace green channel by yellow ER channel
            rgby[:, :, 1] = create_green_channel_albumentations(rgby[:, :, 3], self.ermt_transform)
            image_level_label[6] = 1
        elif tmp < self.p_er + self.p_mt:
            # Replace green channel by red Microtubule channel
            rgby[:, :, 1] = create_green_channel_albumentations(rgby[:, :, 0], self.ermt_transform)
            image_level_label[10] = 1
        else:
            # Default case: use original image & original label
            for value in label.split("|"):
                image_level_label[int(value)] = 1

        example = {"image": rgby, "mask": cell_mask}
        example = self.transform(**example)

        # rgby = np.moveaxis(example["image"], 2, 0)
        rgby = example["image"]
        cell_mask = example["mask"]

        if self.pixel_wise_heavier_aug:
            rgby_heavier_aug = self.pixel_wise_heavier_aug(image=rgby)["image"]
        else:
            rgby_heavier_aug = rgby.copy()

        if np.random.rand() < self.drop_cell_p:
            num_instance = cell_mask.max()
            if num_instance >= 2:
                # The number of instances to drop is determined by a uniform random number of [1, number of instances - 1]
                num_drop = np.random.randint(1, num_instance)
                drop_instance_indices = np.random.choice(num_instance, size=num_drop, replace=False) + 1  # 1-origin
                drop_mask = np.isin(cell_mask, drop_instance_indices)
                kernel = np.ones((3, 3), dtype=np.uint8)  # Also erase boundaries
                drop_mask = cv2.dilate(drop_mask.astype(np.uint8), kernel, iterations=1).astype(np.bool8)
                rgby_heavier_aug[drop_mask] = 0
                cell_mask[drop_mask] = 0

        rgby = np.moveaxis(rgby, 2, 0)
        rgby_heavier_aug = np.moveaxis(rgby_heavier_aug, 2, 0)

        ret = [rgby, rgby_heavier_aug, image_level_label, cell_mask]

        if self.pseudo_cell_level_labels is not None:
            cell_level_label = self.pseudo_cell_level_labels[index]
            ret.append(cell_level_label)

        return tuple(ret)


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_ids: Sequence[str],
        labels: Sequence[str],
        image_directory: str,
        transform: A.Compose,
    ) -> None:
        self.image_ids = image_ids
        self.labels = labels
        self.image_directory = DirectoryInZip(image_directory)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        label = self.labels[index]

        rgby = read_rgby(self.image_directory, image_id, channel_last=True).astype(np.float32)

        image_level_label = np.zeros(19, dtype=np.float32)
        for value in label.split("|"):
            image_level_label[int(value)] = 1

        example = {"image": rgby}
        example = self.transform(**example)

        rgby = np.moveaxis(example["image"], 2, 0)
        return rgby, image_level_label


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_ids: Sequence[str],
        image_directory: str,
        cell_mask_directory: str,
        transform: A.Compose,
    ) -> None:
        self.image_ids = image_ids
        self.image_directory = DirectoryInZip(image_directory)
        self.cell_mask_directory = DirectoryInZip(cell_mask_directory)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        rgby = read_rgby(self.image_directory, image_id, channel_last=True).astype(np.float32)
        cell_mask = read_gray(self.cell_mask_directory, f"{image_id}.png")

        example = {"image": rgby, "mask": cell_mask}
        example = self.transform(**example)

        rgby = np.moveaxis(example["image"], 2, 0)
        cell_mask = example["mask"]
        return rgby, cell_mask


class SMPBackbone(nn.Module):
    def __init__(self, arch: str, pretrained: bool, in_channels: int):
        super().__init__()
        arch_class, encoder_name = arch[: arch.index("-")], arch[arch.index("-") + 1 :]
        encoder_weights = "imagenet" if pretrained else None
        self.segmentation_model = getattr(smp, arch_class)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        assert isinstance(self.segmentation_model, smp.base.SegmentationModel)
        self.num_features = self.segmentation_model.segmentation_head[0].in_channels
        del self.segmentation_model.segmentation_head

    def forward_features(self, x) -> Tensor:
        features = self.segmentation_model.encoder(x)
        return self.segmentation_model.decoder(*features)


class CosineClassifier(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, bias=False)
        self.s = nn.Parameter(torch.ones(1) * 20.0)

    def forward(self, input: Tensor) -> Tensor:
        input = self.s * F.normalize(input, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return self._conv_forward(input, weight)


class CAMClassifier(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        pretrained: bool = True,
        out_channels: int = 19,
    ) -> None:
        super().__init__()

        if config.arch.startswith("smp-"):
            self.backbone = SMPBackbone(arch=config.arch[4:], pretrained=pretrained, in_channels=4)
        else:
            self.backbone = timm.create_model(config.arch, pretrained=pretrained, num_classes=0, in_chans=4)

        self.use_cosine_classifier = config.use_cosine_classifier
        if config.use_cosine_classifier:
            self.classifier = CosineClassifier(self.backbone.num_features, out_channels, 1)
        else:
            self.classifier = nn.Conv2d(self.backbone.num_features, out_channels, 1, bias=False)

        self.norm: Optional[nn.InstanceNorm2d] = None
        if config.instance_norm_input:
            self.norm = nn.InstanceNorm2d(4)

        if config.remove_maxpool:
            assert isinstance(self.backbone, timm.models.ResNet)
            self.backbone.maxpool = nn.Identity()

        if config.remove_first_stride2:
            for module in self.backbone.modules():
                if isinstance(module, nn.Conv2d):
                    module.stride = (1, 1)
                    break

    def forward(self, x, with_cam: bool = False):
        if self.norm is not None:
            x = self.norm(x)

        features = self.backbone.forward_features(x)

        if with_cam:
            cam = self.classifier(features)
            logits = F.adaptive_avg_pool2d(cam, 1)
            logits = logits.view(*logits.shape[:2])
            return logits, cam
        else:
            if self.use_cosine_classifier:
                # For cosine_classifier, the order of pool and linear will change the result, so always do linear first.
                logits = F.adaptive_avg_pool2d(self.classifier(features), 1)
            else:
                # If cam is not needed, pooling first is less computationally expensive.
                logits = self.classifier(F.adaptive_avg_pool2d(features, 1))
            logits = logits.view(*logits.shape[:2])
            return logits


class GetCAMConsistencyTrainingUpdateFn:
    def __init__(
        self,
        max_alpha: float,
        alpha_schedule: float,
        shear_range_abs: float,
        angle_limit: float,
        scale_limit: float,
        cutmix_p: float,
        cutmix_p_in_heavier_aug: float,
        cutmix_repeat: int,
        consistency_loss_criteria: str,
        consistency_loss_targets: str,
        hard_label_threshold: Optional[float],
        weight_image_level_class_loss: float,
        edge_area_threshold: int,
        center_area_threshold: int,
    ) -> None:
        self.max_alpha = max_alpha
        self.alpha_schedule = alpha_schedule
        self.shear_range_abs = shear_range_abs
        self.angle_limit = angle_limit
        self.scale_limit = scale_limit
        self.cutmix_p = cutmix_p
        self.cutmix_p_in_heavier_aug = cutmix_p_in_heavier_aug
        self.consistency_loss_criteria = consistency_loss_criteria
        self.consistency_loss_targets = consistency_loss_targets
        self.cutmix_repeat = cutmix_repeat
        self.hard_label_threshold = hard_label_threshold
        self.weight_image_level_class_loss = weight_image_level_class_loss
        self.edge_area_threshold = edge_area_threshold
        self.center_area_threshold = center_area_threshold

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable, torch.nn.Module],
        device: Optional[Union[str, torch.device]],
        label_indices: Sequence[int],
        non_blocking: bool = False,
    ) -> Callable:
        def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
            model.train()
            optimizer.zero_grad()

            batch = [convert_tensor(v, device=device, non_blocking=non_blocking) for v in batch]
            x, x_heavier_aug, image_level_label, cell_mask = batch[:4]

            B, _, H, W = x.shape

            if len(batch) == 5:
                cell_level_label = batch[4]
            else:
                cell_level_label = None

            # Augmentation that shifts the positional relationship
            angle = np.random.uniform(0, self.angle_limit)
            shear_x = np.random.uniform(-self.shear_range_abs, self.shear_range_abs)
            shear_y = np.random.uniform(-self.shear_range_abs, self.shear_range_abs)
            scale = 1.0 + np.random.uniform(-self.scale_limit, self.scale_limit)

            def _affine(image: Tensor, resample=PIL.Image.BILINEAR) -> Tensor:
                return torchvision.transforms.functional.affine(
                    image,
                    angle=angle,
                    translate=(0, 0),
                    scale=scale,
                    shear=(shear_x, shear_y),
                    resample=resample,
                )

            x_heavier_aug = _affine(x_heavier_aug)
            cell_mask = _affine(cell_mask, resample=PIL.Image.NEAREST)

            # CutMix
            use_cutmix = np.random.rand() < self.cutmix_p
            use_cutmix_in_heavier_aug = np.random.rand() < self.cutmix_p_in_heavier_aug

            if use_cutmix:
                cutmix_params = []
                rand_index = torch.randperm(B).cuda()
                for _ in range(self.cutmix_repeat):
                    x, *cutmix_param = apply_cutmix(x, rand_index=rand_index)
                    cutmix_params.append(cutmix_param)

                    rand_index = (rand_index + 1) % B  # rotate

            if use_cutmix_in_heavier_aug:
                cutmix_params_heavier_aug = []
                rand_index = torch.randperm(B).cuda()
                for _ in range(self.cutmix_repeat):
                    x_heavier_aug, *cutmix_param_heavier_aug = apply_cutmix(x_heavier_aug, rand_index=rand_index)
                    cutmix_params_heavier_aug.append(cutmix_param_heavier_aug)

                    rand_index = (rand_index + 1) % B  # rotate

            # Compute CAM
            _, cam = model(torch.cat([x, x_heavier_aug], dim=0), with_cam=True)
            cam, cam_heavier_aug = torch.split(cam, B, dim=0)

            # Restore the resolution
            cam = F.interpolate(cam, (H, W), mode="bilinear", align_corners=False)
            cam_heavier_aug = F.interpolate(cam_heavier_aug, (H, W), mode="bilinear", align_corners=False)
            assert cam.shape == cam_heavier_aug.shape

            # Inverse CutMix
            # In normal CutMix, the label side is mixed, but in order to calculate the consistency loss in CAM,
            # the inverse transformation is performed in CAM.
            if use_cutmix:
                for cutmix_param in cutmix_params[::-1]:
                    cam = inverse_cutmix(cam, *cutmix_param)

            if use_cutmix_in_heavier_aug:
                for cutmix_param_heavier_aug in cutmix_params_heavier_aug[::-1]:
                    cam_heavier_aug = inverse_cutmix(cam_heavier_aug, *cutmix_param_heavier_aug)

            # Create a logit by pooling the inverted CAM
            logits = torch.mean(cam, dim=(2, 3))

            # Image level supervised loss
            class_loss = loss_fn(logits, image_level_label)

            # Consistency loss
            if cell_level_label is None:
                # If no pseudo label is given externally, use CAM as the pseudo label.

                # Detach as it is used as an answer
                cam = cam.detach()

                # Apply transformation to cam to align it with cam_heavier_aug
                cam = _affine(cam)

                # Aggregate by cell and apply sigmoid to make a pseudo label
                cell_level_logits, _ = cell_wise_average_pooling(cam, cell_mask)
                cell_level_label = torch.sigmoid(cell_level_logits)
            else:
                # Use a pseudo label given from outside

                # Since sigmoid is taken outside and it seems to be numerically inappropriate to return it
                # to logits, only CE and Focal are supported.
                cell_level_logits = None
                assert self.consistency_loss_criteria in ["CE", "Focal"]

            # Also aggregate cam_heavier_aug by cell
            cell_wise_logits_heavier_aug, pixel_count = cell_wise_average_pooling(cam_heavier_aug, cell_mask)

            # Select the target of the consistency loss calculation
            is_targets = pixel_count > 0
            if self.consistency_loss_targets == "all":
                pass
            elif self.consistency_loss_targets == "positive_only":
                is_targets = is_targets & (image_level_label[:, :, np.newaxis] == 1.0)
            else:
                raise ValueError

            # Harden labels that exceed the threshold
            if self.hard_label_threshold is not None:
                is_targets &= (cell_level_label <= (1.0 - self.hard_label_threshold)) | (
                    self.hard_label_threshold <= cell_level_label
                )
                cell_level_label[cell_level_label <= (1.0 - self.hard_label_threshold)] = 0.0
                cell_level_label[self.hard_label_threshold <= cell_level_label] = 1.0

            # Computation of consistency loss
            if self.consistency_loss_criteria == "CE":
                consistency_loss = F.binary_cross_entropy_with_logits(
                    cell_wise_logits_heavier_aug, cell_level_label, reduction="none"
                )
            elif self.consistency_loss_criteria == "Focal":
                consistency_loss = binary_focal_with_logits(
                    cell_wise_logits_heavier_aug, cell_level_label, reduction="none"
                )
            elif self.consistency_loss_criteria == "L1":
                assert cell_level_logits is not None
                consistency_loss = torch.abs(cell_wise_logits_heavier_aug, cell_level_logits)
            elif self.consistency_loss_criteria == "L2":
                assert cell_level_logits is not None
                consistency_loss = (cell_wise_logits_heavier_aug - cell_level_logits) ** 2
            else:
                raise ValueError

            if self.edge_area_threshold == -1 or self.center_area_threshold == -1:
                consistency_loss = consistency_loss[is_targets].sum() / (is_targets.sum() + 1e-6)
            else:
                loss_weight_on_each_cell = calc_conf_scale_on_batch(
                    cell_mask, self.edge_area_threshold, self.center_area_threshold
                )
                consistency_loss = (consistency_loss[is_targets] * loss_weight_on_each_cell[is_targets]).sum() / (
                    is_targets.sum() + 1e-6
                )

            # Schedule alpha
            iteration = engine.state.iteration
            max_iteration = engine.state.max_epochs * engine.state.epoch_length
            if self.alpha_schedule <= 0.0:
                alpha = self.max_alpha
            else:
                alpha = min(self.max_alpha, self.max_alpha * (iteration / (max_iteration * self.alpha_schedule)))

            loss = self.weight_image_level_class_loss * class_loss + alpha * consistency_loss

            loss.backward()
            optimizer.step()

            return {
                "loss": loss.item(),
                "class_loss": class_loss.item(),
                "consistency_loss": consistency_loss.item(),
                "alpha": alpha,
            }

        return _update


def make_prediction(
    model: nn.Module,
    dataset: InferenceDataset,
    batch_size: int,
    num_workers: int = 4,
    device: str = "cuda",
):
    model.to(device)
    model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    predictions = np.empty(len(dataset), dtype=object)
    instance_indices = np.empty(len(dataset), dtype=object)
    for i in range(len(dataset)):
        predictions[i] = []
        instance_indices[i] = []

    image_level_predictions = []

    with torch.no_grad():

        def _get_non_ref_array(x):
            return x.detach().cpu().numpy().copy()

        start = 0
        for batch in tqdm(data_loader):

            rgby, cell_mask = [convert_tensor(v, device=device, non_blocking=False) for v in batch]
            image_level_logits, cam = model(rgby, with_cam=True)

            cam = F.interpolate(cam, cell_mask.shape[-2:])
            cell_level_pred, pixel_count = cell_wise_average_pooling(cam, cell_mask)

            cell_level_pred = torch.sigmoid(cell_level_pred)

            assert cell_level_pred.shape == pixel_count.shape  # == (B, C, N)
            cell_level_pred = _get_non_ref_array(cell_level_pred)
            pixel_count = _get_non_ref_array(pixel_count)

            # cell_mask is the same for all classes
            assert (pixel_count[:, :1, :] == pixel_count).all()
            pixel_count = pixel_count[:, 0, :]

            indices = np.arange(start, start + rgby.shape[0])
            start += rgby.shape[0]

            for instance_index in range(cell_level_pred.shape[-1]):
                appear = pixel_count[:, instance_index] > 0

                sub_y = cell_level_pred[appear, :, instance_index]
                sub_indices = indices[appear]

                for j, index in enumerate(sub_indices):
                    predictions[index].append(sub_y[j, :])
                    instance_indices[index].append(instance_index + 1)

            image_level_pred = torch.sigmoid(image_level_logits)
            image_level_predictions.append(_get_non_ref_array(image_level_pred))

    concat_predictions = np.concatenate([np.asarray(vs).reshape(len(vs), 19) for vs in predictions], axis=0)
    concat_instance_indices = np.concatenate(instance_indices)
    sizes = np.array([len(vs) for vs in predictions])

    return concat_predictions, concat_instance_indices, sizes, np.concatenate(image_level_predictions, axis=0)


def make_prediction_with_d4_tta(
    model: nn.Module,
    dataset: InferenceDataset,
    batch_size: int,
    num_workers: int = 4,
    device: str = "cuda",
):
    aug_candidates = [A.VerticalFlip(p=1.0), A.HorizontalFlip(p=1.0), As.FixedFactorRandomRotate90(p=1.0, factor=1)]
    use_augs_gen = itertools.product([True, False], repeat=len(aug_candidates))

    image_level_predictions_list = []
    concat_predictions_list = []
    concat_instance_indices, sizes = None, None

    for use_augs in use_augs_gen:
        aug_list = [aug for use, aug in zip(use_augs, aug_candidates) if use]
        transform = A.Compose(aug_list + list(dataset.transform.transforms))

        augmented_dataset = InferenceDataset(
            dataset.image_ids, str(dataset.image_directory), str(dataset.cell_mask_directory), transform
        )

        concat_predictions, concat_instance_indices_, sizes_, image_level_predictions = make_prediction(
            model, augmented_dataset, batch_size, num_workers, device
        )

        concat_predictions_list.append(concat_predictions)
        image_level_predictions_list.append(image_level_predictions)

        if sizes is None:
            assert concat_instance_indices is None
            sizes = sizes_
            concat_instance_indices = concat_instance_indices_
        else:
            assert concat_instance_indices is not None
            assert (sizes == sizes_).all()
            assert (concat_instance_indices == concat_instance_indices_).all()

    return (
        np.stack(concat_predictions_list, axis=-1),
        concat_instance_indices,
        sizes,
        np.stack(image_level_predictions_list, axis=-1),
    )


def make_image_level_prediction_with_d4_tta(
    model: nn.Module,
    dataset: InferenceDataset,
    batch_size: int,
    num_workers: int = 4,
    device: str = "cuda",
):
    aug_candidates = [A.VerticalFlip(p=1.0), A.HorizontalFlip(p=1.0), As.FixedFactorRandomRotate90(p=1.0, factor=1)]
    use_augs_gen = itertools.product([True, False], repeat=len(aug_candidates))

    pred_list = []

    for use_augs in use_augs_gen:
        aug_list = [aug for use, aug in zip(use_augs, aug_candidates) if use]
        transform = A.Compose(aug_list + list(dataset.transform.transforms))

        augmented_dataset = InferenceDataset(
            dataset.image_ids, str(dataset.image_directory), str(dataset.cell_mask_directory), transform
        )

        pred = somen.pytorch_utility.predict(model, augmented_dataset, batch_size, num_workers, device)
        pred_list.append(pred)

    return np.stack(pred_list, axis=-1)


def _make_pseudo_cell_level_label(pseudo_cell_level_label_dir: Path, public_hpa: bool = False) -> np.ndarray:
    if public_hpa:
        concat_predictions = somen.file_io.load_array(pseudo_cell_level_label_dir / "predictions_valid_pub_hpa.h5")
        concat_instance_indices = somen.file_io.load_array(
            pseudo_cell_level_label_dir / "instance_indices_valid_pub_hpa.h5"
        )
        sizes = somen.file_io.load_array(pseudo_cell_level_label_dir / "sizes_valid_pub_hpa.h5")
    else:
        concat_predictions = somen.file_io.load_array(pseudo_cell_level_label_dir / "predictions_valid.h5")
        concat_instance_indices = somen.file_io.load_array(pseudo_cell_level_label_dir / "instance_indices_valid.h5")
        sizes = somen.file_io.load_array(pseudo_cell_level_label_dir / "sizes_valid.h5")

    # concat_predictions: (num_total_cell, class_num) or (num_total_cell, class_num, num_tta)
    if concat_predictions.ndim == 3:
        concat_predictions = concat_predictions.mean(axis=-1)

    assert (concat_instance_indices == concat_instance_indices.astype(np.int64)).all()
    concat_instance_indices = concat_instance_indices.astype(np.int64)

    sections = np.cumsum(sizes[:-1])

    cell_level_label = np.zeros((sizes.shape[0], concat_predictions.shape[1], 255))
    for i, (pred, instance_indices) in enumerate(
        zip(np.split(concat_predictions, sections), np.split(concat_instance_indices, sections))
    ):
        # instance_indices is 1-origin, so it should be -1
        cell_level_label[i, :, instance_indices - 1] = pred

    return cell_level_label


def _read_public_hpa_df(csv_path: str = "data/input/kaggle_2021.tsv", drop_na: bool = True):
    df_pub_hpa = pd.read_csv(csv_path)
    df_pub_hpa["ID"] = df_pub_hpa["Image"].map(lambda x: x.split("/")[-1])
    df_pub_hpa = df_pub_hpa.dropna()
    df_pub_hpa = df_pub_hpa.loc[~df_pub_hpa["in_trainset"]].reset_index(drop=True)
    df_pub_hpa["Label_name"] = df_pub_hpa["Label"]  # backup
    df_pub_hpa["Label"] = df_pub_hpa["Label_idx"]
    return df_pub_hpa


def train(
    config_path: str,
    *overrides: Sequence[str],
    use_wandb: bool = False,
    resume: bool = False,
    num_workers: int = 4,
    device: str = "cuda",
    benchmark: bool = False,
    cprofile: bool = False,
    csv_path: str = "data/input/train.csv",
    local_rank: Optional[int] = None,
) -> None:
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, Path(config_path), overrides)
    working_dir = Path(f"data/working/consistency_training/{Path(config_path).stem}/{config.fold_index}/")

    distributed = local_rank is not None
    if distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    is_main_node = (not distributed) or (torch.distributed.get_rank() == 0)
    if is_main_node:
        somen.file_io.save_yaml_from_dataclass(config, working_dir / "config.yaml")
        print(asdict(config))

    ct_config = config.consistency_training
    get_update_fn = GetCAMConsistencyTrainingUpdateFn(
        max_alpha=ct_config.max_alpha,
        alpha_schedule=ct_config.alpha_schedule,
        shear_range_abs=ct_config.shear_range_abs,
        angle_limit=ct_config.angle_limit,
        scale_limit=ct_config.scale_limit,
        cutmix_p=ct_config.cutmix_p,
        cutmix_p_in_heavier_aug=ct_config.cutmix_p_in_heavier_aug,
        cutmix_repeat=ct_config.cutmix_repeat,
        consistency_loss_criteria=ct_config.consistency_loss_criteria,
        consistency_loss_targets=ct_config.consistency_loss_targets,
        hard_label_threshold=ct_config.hard_label_threshold,
        weight_image_level_class_loss=ct_config.weight_image_level_class_loss,
        edge_area_threshold=ct_config.edge_area_threshold,
        center_area_threshold=ct_config.center_area_threshold,
    )

    df = pd.read_csv(csv_path)
    train_indices, valid_indices = get_folds(df, config.n_folds)[config.fold_index]

    pseudo_cell_level_label = None
    if config.train_dataset.pseudo_cell_level_label_dir is not None:
        pseudo_cell_level_label = _make_pseudo_cell_level_label(config.train_dataset.pseudo_cell_level_label_dir)
        assert len(pseudo_cell_level_label) == len(df)

    train_dataset = TrainingDataset(
        image_ids=df.loc[train_indices, "ID"].to_numpy(),
        labels=df.loc[train_indices, "Label"].to_numpy(),
        image_directory=config.train_dataset.image_directory,
        cell_mask_directory=config.train_dataset.cell_mask_directory,
        transform=A.from_dict(config.train_transform),
        pixel_wise_heavier_aug=(
            None if ct_config.pixel_wise_heavier_aug is None else A.from_dict(ct_config.pixel_wise_heavier_aug)
        ),
        drop_cell_p=ct_config.drop_cell_p,
        pseudo_cell_level_labels=None if pseudo_cell_level_label is None else pseudo_cell_level_label[train_indices],
        p_er=ct_config.p_er,
        p_mt=ct_config.p_mt,
        ermt_transform=A.from_dict(ct_config.ermt_transform) if ct_config.ermt_transform is not None else None,
    )
    valid_dataset = ValidationDataset(
        image_ids=df.loc[valid_indices, "ID"].to_numpy(),
        labels=df.loc[valid_indices, "Label"].to_numpy(),
        image_directory=config.train_dataset.image_directory,
        transform=A.from_dict(config.valid_transform),
    )

    if config.public_hpa_dataset is not None:
        df_pub_hpa = _read_public_hpa_df()
        train_indices_pub_hpa, valid_indices_pub_hpa = get_folds(df_pub_hpa, config.n_folds)[config.fold_index]

        pseudo_cell_level_label_pub_hpa = None
        if config.train_dataset.pseudo_cell_level_label_dir is not None:
            pseudo_cell_level_label_pub_hpa = _make_pseudo_cell_level_label(
                config.public_hpa_dataset.pseudo_cell_level_label_dir,
                public_hpa=True,
            )
            assert len(pseudo_cell_level_label_pub_hpa) == len(df_pub_hpa)

        train_dataset_pub_hpa = TrainingDataset(
            image_ids=df_pub_hpa.loc[train_indices_pub_hpa, "ID"].to_numpy(),
            labels=df_pub_hpa.loc[train_indices_pub_hpa, "Label"].to_numpy(),
            image_directory=config.public_hpa_dataset.image_directory,
            cell_mask_directory=config.public_hpa_dataset.cell_mask_directory,
            transform=A.from_dict(config.train_transform),
            pixel_wise_heavier_aug=(
                None if ct_config.pixel_wise_heavier_aug is None else A.from_dict(ct_config.pixel_wise_heavier_aug)
            ),
            drop_cell_p=ct_config.drop_cell_p,
            pseudo_cell_level_labels=(
                None
                if pseudo_cell_level_label_pub_hpa is None
                else pseudo_cell_level_label_pub_hpa[train_indices_pub_hpa]
            ),
            p_er=ct_config.p_er,
            p_mt=ct_config.p_mt,
            ermt_transform=A.from_dict(ct_config.ermt_transform) if ct_config.ermt_transform is not None else None,
        )
        valid_dataset_pub_hpa = ValidationDataset(
            image_ids=df_pub_hpa.loc[valid_indices_pub_hpa, "ID"].to_numpy(),
            labels=df_pub_hpa.loc[valid_indices_pub_hpa, "Label"].to_numpy(),
            image_directory=config.public_hpa_dataset.image_directory,
            transform=A.from_dict(config.valid_transform),
        )

        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_pub_hpa])
        valid_dataset = torch.utils.data.ConcatDataset([valid_dataset, valid_dataset_pub_hpa])

    model = CAMClassifier(
        config.model,
        pretrained=(config.model.pretrained is None),  # If None, use ImageNet pretrained weights
        out_channels=19,
    )
    if config.model.pretrained is not None:
        pretrained = config.model.pretrained.format(fold_index=config.fold_index)
        print(f"Loading pretrained model: {pretrained}")
        print(model.load_state_dict(torch.load(pretrained, map_location="cpu")))

    if config.optim.objective == "bce":
        loss_func = F.binary_cross_entropy_with_logits
    elif config.optim.objective == "focal":
        loss_func = binary_focal_with_logits
    else:
        raise ValueError

    params = {
        "objective": loss_func,
        "get_update_fn": get_update_fn,
        "optimizer": config.optim.optimizer,
        "optimizer_params": config.optim.optimizer_params,
        "nb_epoch": config.optim.nb_epoch,
        "batch_size": config.optim.batch_size,
        "device": device,
        "num_workers": num_workers,
        "resume": resume,
        "benchmark_mode": benchmark,
        "enable_cprofile": cprofile,
        "trainer_snapshot_n_saved": 1,
        "metric": [("bce", F.binary_cross_entropy_with_logits), ("focal", binary_focal_with_logits), ("map", mAP)],
        "batch_eval": True,
        "maximize": [False, False, True],
        "local_rank": local_rank,
        "lr_scheduler": config.optim.lr_scheduler,
        "lr_scheduler_params": config.optim.lr_scheduler_params,
        "find_unused_parameters": True,
        "sampler_seed": config.optim.sampler_seed,
    }
    ext_extensions = []

    if use_wandb:
        if is_main_node:
            wandb_init(
                out=working_dir,
                resume_from_id=resume,
                entity="kaggle-hpa-pfcell",
                project="p20-kaggle-hpa",
                group=Path(config_path).stem,
                name=f"fold-{config.fold_index}",
                config=asdict(config),
                config_exclude_keys=["train_transform", "test_transform", "valid_transform"],
            )
        ext_extensions.append(WandbReporter())

    if is_main_node:
        print("Start training")

    try:
        somen.pytorch_utility.train(
            model=model,
            params=params,
            train_set=train_dataset,
            valid_sets=[valid_dataset],
            working_dir=working_dir,
            load_best=False,
            ext_extensions=ext_extensions,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if is_main_node:
            torch.save(model.state_dict(), working_dir / "final.pth")


def predict_valid(
    config_path: str,
    *overrides: Sequence[str],
    num_workers: int = 4,
    device: str = "cuda",
    snapshot_name: str = "final.pth",
    csv_path: str = "data/input/train.csv",
) -> None:
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, Path(config_path), overrides)
    working_dir = Path(f"data/working/consistency_training/{Path(config_path).stem}/{config.fold_index}/")

    print(asdict(config))

    df = pd.read_csv(csv_path)
    valid_indices = get_folds(df, config.n_folds)[config.fold_index][1]

    valid_dataset = InferenceDataset(
        image_ids=df.loc[valid_indices, "ID"].to_numpy(),
        image_directory=config.train_dataset.image_directory,
        cell_mask_directory=config.train_dataset.cell_mask_directory,
        transform=A.from_dict(config.test_transform),
    )
    model = CAMClassifier(
        config.model,
        pretrained=False,
        out_channels=19,
    )
    print(model.load_state_dict(torch.load(str(working_dir / snapshot_name), map_location="cpu")))

    concat_predictions, concat_instance_indices, sizes, image_level_predictions = make_prediction_with_d4_tta(
        model, valid_dataset, config.optim.batch_size * 2, num_workers, device
    )
    somen.file_io.save_array(concat_predictions, working_dir / "predictions_valid.h5")
    somen.file_io.save_array(concat_instance_indices, working_dir / "instance_indices_valid.h5")
    somen.file_io.save_array(sizes, working_dir / "sizes_valid.h5")
    somen.file_io.save_array(image_level_predictions, working_dir / "image_level_predictions_valid.h5")

    if config.public_hpa_dataset is not None:
        df_pub_hpa = _read_public_hpa_df()
        valid_indices_pub_hpa = get_folds(df_pub_hpa, config.n_folds)[config.fold_index][1]

        valid_dataset_pub_hpa = InferenceDataset(
            image_ids=df_pub_hpa.loc[valid_indices_pub_hpa, "ID"].to_numpy(),
            image_directory=config.public_hpa_dataset.image_directory,
            cell_mask_directory=config.public_hpa_dataset.cell_mask_directory,
            transform=A.from_dict(config.test_transform),
        )

        concat_predictions, concat_instance_indices, sizes, image_level_predictions = make_prediction_with_d4_tta(
            model, valid_dataset_pub_hpa, config.optim.batch_size * 2, num_workers, device
        )
        somen.file_io.save_array(concat_predictions, working_dir / "predictions_valid_pub_hpa.h5")
        somen.file_io.save_array(concat_instance_indices, working_dir / "instance_indices_valid_pub_hpa.h5")
        somen.file_io.save_array(sizes, working_dir / "sizes_valid_pub_hpa.h5")
        somen.file_io.save_array(image_level_predictions, working_dir / "image_level_predictions_valid_pub_hpa.h5")


def predict_test(
    config_path: str,
    *overrides: Sequence[str],
    num_workers: int = 4,
    device: str = "cuda",
    snapshot_name: str = "final.pth",
) -> None:
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, Path(config_path), overrides)
    working_dir = Path(f"data/working/consistency_training/{Path(config_path).stem}/{config.fold_index}/")

    print(asdict(config))

    assert config.test_dataset.orig_size_cell_mask_directory is not None

    image_ids = np.unique(
        [filename.split("_")[0] for filename in DirectoryInZip(config.test_dataset.image_directory).listdir()]
    )
    image_ids = sorted(image_ids)

    test_dataset = InferenceDataset(
        image_ids=image_ids,
        image_directory=config.test_dataset.image_directory,
        cell_mask_directory=config.test_dataset.cell_mask_directory,
        transform=A.from_dict(config.test_transform),
    )
    model = CAMClassifier(
        config.model,
        pretrained=False,
        out_channels=19,
    )
    print(model.load_state_dict(torch.load(str(working_dir / snapshot_name), map_location="cpu")))

    concat_predictions, concat_instance_indices, sizes, image_level_predictions = make_prediction_with_d4_tta(
        model, test_dataset, config.optim.batch_size * 2, num_workers, device
    )
    somen.file_io.save_array(concat_predictions, working_dir / "predictions_test.h5")
    somen.file_io.save_array(concat_instance_indices, working_dir / "instance_indices_test.h5")
    somen.file_io.save_array(sizes, working_dir / "sizes_test.h5")
    somen.file_io.save_array(image_level_predictions, working_dir / "image_level_predictions_test.h5")


def concat_valid_predictions(
    config_path: str,
    *overrides: Sequence[str],
    csv_path: str = "data/input/train.csv",
) -> None:
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, Path(config_path), overrides)

    def _concat_valid_predictions(public_hpa: bool) -> None:
        if public_hpa:
            df = _read_public_hpa_df()
        else:
            df = pd.read_csv(csv_path)

        folds = get_folds(df, config.n_folds)

        data = []
        for fold_index in range(config.n_folds):
            working_dir = Path(f"data/working/consistency_training/{Path(config_path).stem}/{fold_index}/")

            if public_hpa:
                concat_predictions = somen.file_io.load_array(working_dir / "predictions_valid_pub_hpa.h5")
                concat_instance_indices = somen.file_io.load_array(working_dir / "instance_indices_valid_pub_hpa.h5")
                sizes = somen.file_io.load_array(working_dir / "sizes_valid_pub_hpa.h5")
                image_level_predictions = somen.file_io.load_array(
                    working_dir / "image_level_predictions_valid_pub_hpa.h5"
                )
            else:
                concat_predictions = somen.file_io.load_array(working_dir / "predictions_valid.h5")
                concat_instance_indices = somen.file_io.load_array(working_dir / "instance_indices_valid.h5")
                sizes = somen.file_io.load_array(working_dir / "sizes_valid.h5")
                image_level_predictions = somen.file_io.load_array(working_dir / "image_level_predictions_valid.h5")

            sections = np.cumsum(sizes[:-1])
            predictions = np.asarray(np.split(concat_predictions, sections), dtype=object)
            instance_indices = np.asarray(np.split(concat_instance_indices, sections), dtype=object)

            data.append((predictions, instance_indices, image_level_predictions))

        data = [np.concatenate([e[j] for e in data], axis=0) for j in range(3)]
        permutation = np.concatenate([valid_indices for _, valid_indices in folds], axis=0)
        inv_permutation = np.argsort(permutation)
        data = [e[inv_permutation] for e in data]

        concat_predictions = np.concatenate(data[0], axis=0)
        concat_instance_indices = np.concatenate(data[1], axis=0)
        image_level_predictions = data[2]
        sizes = np.asarray([len(e) for e in data[0]])

        out_dir = Path(f"data/working/consistency_training/{Path(config_path).stem}/")

        if public_hpa:
            somen.file_io.save_array(concat_predictions, out_dir / "predictions_valid_pub_hpa.h5")
            somen.file_io.save_array(concat_instance_indices, out_dir / "instance_indices_valid_pub_hpa.h5")
            somen.file_io.save_array(sizes, out_dir / "sizes_valid_pub_hpa.h5")
            somen.file_io.save_array(image_level_predictions, out_dir / "image_level_predictions_valid_pub_hpa.h5")
        else:
            somen.file_io.save_array(concat_predictions, out_dir / "predictions_valid.h5")
            somen.file_io.save_array(concat_instance_indices, out_dir / "instance_indices_valid.h5")
            somen.file_io.save_array(sizes, out_dir / "sizes_valid.h5")
            somen.file_io.save_array(image_level_predictions, out_dir / "image_level_predictions_valid.h5")

    _concat_valid_predictions(public_hpa=False)
    _concat_valid_predictions(public_hpa=True)


if __name__ == "__main__":
    setup_forkserver()
    fire.Fire(
        {
            "train": train,
            "predict_valid": predict_valid,
            "predict_test": predict_test,
            "concat_valid_predictions": concat_valid_predictions,
        }
    )
