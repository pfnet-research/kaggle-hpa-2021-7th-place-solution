import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import blosc
import fire
import numpy as np
import pandas as pd
import PIL
import timm
import torch
import torch.nn.functional as F
import torchvision
from ignite.engine import Engine
from ignite.utils import convert_tensor
from rich import print
from torch import Tensor, nn

import somen
from hpa.cutmix import apply_cutmix
from hpa.metrics import binary_focal_with_logits, mAP
from hpa.validation import get_folds
from somen.pfio_utility import DirectoryInZip, setup_forkserver
from somen.pytorch_utility.d4_transforms import D4
from somen.pytorch_utility.extensions.wandb import WandbReporter, wandb_init


@dataclass
class ModelConfig:
    arch: str
    instance_norm_input: bool = True
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
class SoftPseudoLabelingConfig:
    label_dir: Path


@dataclass
class DatasetConfig:
    cell_image_directory: str
    orig_size_cell_mask_directory: Optional[str]
    soft_pseudo_labeling_config: Optional[SoftPseudoLabelingConfig]


@dataclass
class AffineDataAugmentationConfig:
    shear_range_abs: float
    angle_limit: float
    scale_limit: float


@dataclass
class TrainingConfig:
    n_folds: int
    fold_index: int
    model: ModelConfig
    optim: OptimizationConfig
    train_dataset: DatasetConfig
    public_hpa_dataset: Optional[DatasetConfig]
    test_dataset: DatasetConfig
    affine_aug: Optional[AffineDataAugmentationConfig]
    exclude_bad_examples: float = 0.0
    cutmix_p: float = 0.0


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_ids: Sequence[str],
        labels: Sequence[str],
        cell_image_directory: str,
        instance_indices_by_id: Mapping[str, Sequence[int]],
        soft_label: Optional[np.ndarray],
    ) -> None:
        self.image_ids = image_ids
        self.cell_image_directory = DirectoryInZip(cell_image_directory)
        self.instance_indices_by_id = instance_indices_by_id

        self.binary_labels = np.zeros((len(labels), 19), dtype=np.float32)
        for i, label in enumerate(labels):
            for value in label.split("|"):
                self.binary_labels[i, int(value)] = 1

        self.image_id_indices = []
        self.instance_indices = []
        for i, image_id in enumerate(image_ids):
            for instance_index in instance_indices_by_id[image_id]:
                self.image_id_indices.append(i)
                self.instance_indices.append(instance_index)

        if soft_label is not None:
            assert len(soft_label) == len(self.image_id_indices)
        self.soft_label = soft_label

    def __len__(self) -> int:
        return len(self.image_id_indices)

    def __getitem__(self, index: int):
        image_id_index = self.image_id_indices[index]
        instance_index = self.instance_indices[index]

        image_id = self.image_ids[image_id_index]
        if self.soft_label is None:
            label = self.binary_labels[image_id_index]
        else:
            label = self.soft_label[index]

        with self.cell_image_directory.open(f"{image_id}_{instance_index}.blosc", "rb") as fp:
            image = blosc.unpack_array(fp.read())

        return image.astype(np.float32), label


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_ids: Sequence[str],
        cell_image_directory: str,
        instance_indices_by_id: Mapping[str, Sequence[int]],
    ) -> None:
        self.image_ids = image_ids
        self.cell_image_directory = DirectoryInZip(cell_image_directory)
        self.instance_indices_by_id = instance_indices_by_id

        self.image_id_indices = []
        self.instance_indices = []
        self.sizes = []
        for i, image_id in enumerate(image_ids):
            for instance_index in instance_indices_by_id[image_id]:
                self.image_id_indices.append(i)
                self.instance_indices.append(instance_index)
            self.sizes.append(len(instance_indices_by_id[image_id]))

    def __len__(self) -> int:
        return len(self.image_id_indices)

    def __getitem__(self, index: int):
        image_id_index = self.image_id_indices[index]
        instance_index = self.instance_indices[index]

        image_id = self.image_ids[image_id_index]

        with self.cell_image_directory.open(f"{image_id}_{instance_index}.blosc", "rb") as fp:
            image = blosc.unpack_array(fp.read())

        return (image.astype(np.float32),)


class CosineClassifier(nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, bias=False)
        self.s = nn.Parameter(torch.ones(1) * 20.0)

    def forward(self, input: Tensor) -> Tensor:
        input = self.s * F.normalize(input, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return F.linear(input, weight)


class CellCropClassifier(nn.Sequential):
    def __init__(self, config: ModelConfig, pretrained: bool = True, out_channels: int = 19) -> None:
        modules: Sequence[nn.Module] = []
        if config.instance_norm_input:
            modules.append(nn.InstanceNorm2d(8))
        modules.append(timm.create_model(config.arch, pretrained=pretrained, num_classes=out_channels, in_chans=8))
        super().__init__(*modules)

        if config.use_cosine_classifier:
            # Override Linear with CosineClassifier
            base_model = self[-1]
            if isinstance(base_model, timm.models.resnet.ResNet):
                fc_attr = "fc"
            elif isinstance(base_model, timm.models.efficientnet.EfficientNet):
                fc_attr = "classifier"
            else:
                raise RuntimeError
            setattr(base_model, fc_attr, CosineClassifier(base_model.num_features, out_channels))


class GetUpdateFn:
    def __init__(
        self, affine_aug: Optional[AffineDataAugmentationConfig], exclude_bad_examples: float, cutmix_p: float
    ):
        self.affine_aug = affine_aug
        self.exclude_bad_examples = exclude_bad_examples
        self.cutmix_p = cutmix_p

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

            assert len(batch) == 2
            batch = [convert_tensor(v, device=device, non_blocking=non_blocking) for v in batch]
            x, labels = batch

            # Data Augmentation
            x = D4[np.random.randint(8)](x)

            if self.affine_aug is not None:
                angle = np.random.uniform(0, self.affine_aug.angle_limit)
                shear_x = np.random.uniform(-self.affine_aug.shear_range_abs, self.affine_aug.shear_range_abs)
                shear_y = np.random.uniform(-self.affine_aug.shear_range_abs, self.affine_aug.shear_range_abs)
                scale = 1.0 + np.random.uniform(-self.affine_aug.scale_limit, self.affine_aug.scale_limit)
                x = torchvision.transforms.functional.affine(
                    x,
                    angle=angle,
                    translate=(0, 0),
                    scale=scale,
                    shear=(shear_x, shear_y),
                    resample=PIL.Image.BILINEAR,
                )

            if np.random.rand() < self.cutmix_p:
                x, rand_index, bbx1, bby1, bbx2, bby2 = apply_cutmix(x)
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-2] * x.shape[-1]))

                def actual_loss_fn(logits: Tensor, labels: Tensor, reduction: str) -> Tensor:
                    return lam * loss_fn(logits, labels, reduction=reduction) + (1.0 - lam) * loss_fn(
                        logits, labels[rand_index], reduction=reduction
                    )

            else:
                actual_loss_fn = loss_fn

            logits = model(x)

            if self.exclude_bad_examples > 0.0:
                loss = actual_loss_fn(logits, labels, reduction="none")
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss[torch.argsort(loss)[: int(len(loss) * (1 - self.exclude_bad_examples))]])
            else:
                loss = actual_loss_fn(logits, labels, reduction="mean")

            loss.backward()
            optimizer.step()

            return loss.item()

        return _update


def make_prediction_with_d4_tta(
    model: nn.Module,
    dataset: InferenceDataset,
    batch_size: int,
    num_workers: int = 4,
    device: str = "cuda",
):
    model.to(device)
    model.eval()

    concat_predictions_list = []

    for k in range(8):
        (concat_predictions,) = somen.pytorch_utility.predict(
            model=lambda x: torch.sigmoid(model(D4[k](x))),
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            pin_memory=True,
        )
        concat_predictions_list.append(concat_predictions)

    return np.stack(concat_predictions_list, axis=-1)


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
    working_dir = Path(f"data/working/cell_crop/{Path(config_path).stem}/{config.fold_index}/")

    distributed = local_rank is not None
    if distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    is_main_node = (not distributed) or (torch.distributed.get_rank() == 0)
    if is_main_node:
        somen.file_io.save_yaml_from_dataclass(config, working_dir / "config.yaml")
        print(asdict(config))

    def _setup_datasets():
        df = pd.read_csv(csv_path)
        train_indices, valid_indices = get_folds(df, config.n_folds)[config.fold_index]

        cell_image_directory = DirectoryInZip(config.train_dataset.cell_image_directory)
        with cell_image_directory.open("instance_indices_by_id.pkl", "rb") as fp:
            instance_indices_by_id = pickle.load(fp)

        if config.train_dataset.soft_pseudo_labeling_config is not None:
            soft_label = somen.file_io.load_array(
                config.train_dataset.soft_pseudo_labeling_config.label_dir / "predictions_valid.h5"
            )
            if soft_label.ndim == 3:
                soft_label = np.mean(soft_label, axis=-1)
            assert soft_label.ndim == 2
            sizes = somen.file_io.load_array(
                config.train_dataset.soft_pseudo_labeling_config.label_dir / "sizes_valid.h5"
            )
            instance_indices = somen.file_io.load_array(
                config.train_dataset.soft_pseudo_labeling_config.label_dir / "instance_indices_valid.h5"
            )
            assert (
                instance_indices == np.concatenate([instance_indices_by_id[image_id] for image_id in df["ID"]])
            ).all()

            soft_label = np.split(soft_label, np.cumsum(sizes[:-1]))
            soft_label_train = soft_label = np.concatenate([soft_label[i] for i in train_indices], axis=0)
        else:
            soft_label_train = None

        train_dataset = TrainingDataset(
            image_ids=df.loc[train_indices, "ID"].to_numpy(),
            labels=df.loc[train_indices, "Label"].to_numpy(),
            cell_image_directory=config.train_dataset.cell_image_directory,
            instance_indices_by_id=instance_indices_by_id,
            soft_label=soft_label_train,
        )
        valid_dataset = TrainingDataset(
            image_ids=df.loc[valid_indices, "ID"].to_numpy(),
            labels=df.loc[valid_indices, "Label"].to_numpy(),
            cell_image_directory=config.train_dataset.cell_image_directory,
            instance_indices_by_id=instance_indices_by_id,
            soft_label=None,
        )
        return train_dataset, valid_dataset

    def _setup_public_hpa_datasets():
        assert config.public_hpa_dataset is not None
        df_pub_hpa = _read_public_hpa_df()
        train_indices_pub_hpa, valid_indices_pub_hpa = get_folds(df_pub_hpa, config.n_folds)[config.fold_index]

        cell_image_directory = DirectoryInZip(config.public_hpa_dataset.cell_image_directory)
        with cell_image_directory.open("instance_indices_by_id.pkl", "rb") as fp:
            instance_indices_by_id = pickle.load(fp)

        if config.train_dataset.soft_pseudo_labeling_config is not None:
            soft_label = somen.file_io.load_array(
                config.train_dataset.soft_pseudo_labeling_config.label_dir / "predictions_valid_pub_hpa.h5"
            )
            if soft_label.ndim == 3:
                soft_label = np.mean(soft_label, axis=-1)
            assert soft_label.ndim == 2
            sizes = somen.file_io.load_array(
                config.train_dataset.soft_pseudo_labeling_config.label_dir / "sizes_valid_pub_hpa.h5"
            )
            instance_indices = somen.file_io.load_array(
                config.train_dataset.soft_pseudo_labeling_config.label_dir / "instance_indices_valid_pub_hpa.h5"
            )
            assert (
                instance_indices == np.concatenate([instance_indices_by_id[image_id] for image_id in df_pub_hpa["ID"]])
            ).all()

            soft_label = np.split(soft_label, np.cumsum(sizes[:-1]))
            soft_label_train = soft_label = np.concatenate([soft_label[i] for i in train_indices_pub_hpa], axis=0)
        else:
            soft_label_train = None

        train_dataset_pub_hpa = TrainingDataset(
            image_ids=df_pub_hpa.loc[train_indices_pub_hpa, "ID"].to_numpy(),
            labels=df_pub_hpa.loc[train_indices_pub_hpa, "Label"].to_numpy(),
            cell_image_directory=config.public_hpa_dataset.cell_image_directory,
            instance_indices_by_id=instance_indices_by_id,
            soft_label=soft_label_train,
        )
        valid_dataset_pub_hpa = TrainingDataset(
            image_ids=df_pub_hpa.loc[valid_indices_pub_hpa, "ID"].to_numpy(),
            labels=df_pub_hpa.loc[valid_indices_pub_hpa, "Label"].to_numpy(),
            cell_image_directory=config.public_hpa_dataset.cell_image_directory,
            instance_indices_by_id=instance_indices_by_id,
            soft_label=None,
        )
        return train_dataset_pub_hpa, valid_dataset_pub_hpa

    train_dataset, valid_dataset = _setup_datasets()

    if config.public_hpa_dataset is not None:
        train_dataset_pub_hpa, valid_dataset_pub_hpa = _setup_public_hpa_datasets()
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_pub_hpa])
        valid_dataset = torch.utils.data.ConcatDataset([valid_dataset, valid_dataset_pub_hpa])

    get_update_fn = GetUpdateFn(
        affine_aug=config.affine_aug, exclude_bad_examples=config.exclude_bad_examples, cutmix_p=config.cutmix_p
    )

    model = CellCropClassifier(
        config.model,
        pretrained=True,
        out_channels=19,
    )
    if config.model.pretrained is not None:
        pretrained = config.model.pretrained.format(fold_index=config.fold_index)
        print(f"Loading pretrained model: {pretrained}")
        print(model.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False))

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
        "metric": [
            ("cell_level_bce", F.binary_cross_entropy_with_logits),
            ("cell_level_focal", binary_focal_with_logits),
            ("cell_level_map", mAP),
        ],
        "batch_eval": True,
        "maximize": [False, False, True],
        "local_rank": local_rank,
        "lr_scheduler": config.optim.lr_scheduler,
        "lr_scheduler_params": config.optim.lr_scheduler_params,
        "find_unused_parameters": False,
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
                group=f"cell-crop-{Path(config_path).stem}",
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
    csv_path: str = "data/input/train.csv",
) -> None:
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, Path(config_path), overrides)
    working_dir = Path(f"data/working/cell_crop/{Path(config_path).stem}/{config.fold_index}/")

    print(asdict(config))

    def _setup_dataset():
        df = pd.read_csv(csv_path)
        _, valid_indices = get_folds(df, config.n_folds)[config.fold_index]

        cell_image_directory = DirectoryInZip(config.train_dataset.cell_image_directory)
        with cell_image_directory.open("instance_indices_by_id.pkl", "rb") as fp:
            instance_indices_by_id = pickle.load(fp)

        valid_dataset = InferenceDataset(
            image_ids=df.loc[valid_indices, "ID"].to_numpy(),
            cell_image_directory=config.train_dataset.cell_image_directory,
            instance_indices_by_id=instance_indices_by_id,
        )
        return valid_dataset

    def _setup_public_hpa_datasets():
        assert config.public_hpa_dataset is not None
        df_pub_hpa = _read_public_hpa_df()
        _, valid_indices_pub_hpa = get_folds(df_pub_hpa, config.n_folds)[config.fold_index]

        cell_image_directory = DirectoryInZip(config.public_hpa_dataset.cell_image_directory)
        with cell_image_directory.open("instance_indices_by_id.pkl", "rb") as fp:
            instance_indices_by_id = pickle.load(fp)

        valid_dataset_pub_hpa = InferenceDataset(
            image_ids=df_pub_hpa.loc[valid_indices_pub_hpa, "ID"].to_numpy(),
            cell_image_directory=config.public_hpa_dataset.cell_image_directory,
            instance_indices_by_id=instance_indices_by_id,
        )
        return valid_dataset_pub_hpa

    model = CellCropClassifier(
        config.model,
        pretrained=False,
        out_channels=19,
    )
    print(model.load_state_dict(torch.load(str(working_dir / "final.pth"), map_location="cpu")))

    valid_dataset = _setup_dataset()
    concat_predictions = make_prediction_with_d4_tta(
        model, valid_dataset, config.optim.batch_size * 2, num_workers, device
    )

    somen.file_io.save_array(concat_predictions, working_dir / "predictions_valid.h5")
    somen.file_io.save_array(valid_dataset.instance_indices, working_dir / "instance_indices_valid.h5")
    somen.file_io.save_array(valid_dataset.sizes, working_dir / "sizes_valid.h5")

    if config.public_hpa_dataset is not None:
        valid_dataset_pub_hpa = _setup_public_hpa_datasets()

        concat_predictions = make_prediction_with_d4_tta(
            model, valid_dataset_pub_hpa, config.optim.batch_size * 2, num_workers, device
        )
        somen.file_io.save_array(concat_predictions, working_dir / "predictions_valid_pub_hpa.h5")
        somen.file_io.save_array(
            valid_dataset_pub_hpa.instance_indices, working_dir / "instance_indices_valid_pub_hpa.h5"
        )
        somen.file_io.save_array(valid_dataset_pub_hpa.sizes, working_dir / "sizes_valid_pub_hpa.h5")


def predict_test(
    config_path: str,
    *overrides: Sequence[str],
    num_workers: int = 4,
    device: str = "cuda",
) -> None:
    config: TrainingConfig = somen.file_io.load_yaml_as_dataclass(TrainingConfig, Path(config_path), overrides)
    working_dir = Path(f"data/working/cell_crop/{Path(config_path).stem}/{config.fold_index}/")

    cell_image_directory = DirectoryInZip(config.test_dataset.cell_image_directory)
    image_ids = np.unique(
        [filename[: filename.rindex("_")] for filename in cell_image_directory.listdir() if filename.endswith(".blosc")]
    )
    image_ids = sorted(image_ids)

    with cell_image_directory.open("instance_indices_by_id.pkl", "rb") as fp:
        instance_indices_by_id = pickle.load(fp)

    test_dataset = InferenceDataset(
        image_ids=image_ids,
        cell_image_directory=config.test_dataset.cell_image_directory,
        instance_indices_by_id=instance_indices_by_id,
    )

    model = CellCropClassifier(
        config.model,
        pretrained=False,
        out_channels=19,
    )
    print(model.load_state_dict(torch.load(str(working_dir / "final.pth"), map_location="cpu")))

    concat_predictions = make_prediction_with_d4_tta(
        model, test_dataset, config.optim.batch_size * 2, num_workers, device
    )
    somen.file_io.save_array(concat_predictions, working_dir / "predictions_test.h5")
    somen.file_io.save_array(test_dataset.instance_indices, working_dir / "instance_indices_test.h5")
    somen.file_io.save_array(test_dataset.sizes, working_dir / "sizes_test.h5")


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
            working_dir = Path(f"data/working/cell_crop/{Path(config_path).stem}/{fold_index}/")

            if public_hpa:
                concat_predictions = somen.file_io.load_array(working_dir / "predictions_valid_pub_hpa.h5")
                concat_instance_indices = somen.file_io.load_array(working_dir / "instance_indices_valid_pub_hpa.h5")
                sizes = somen.file_io.load_array(working_dir / "sizes_valid_pub_hpa.h5")
            else:
                concat_predictions = somen.file_io.load_array(working_dir / "predictions_valid.h5")
                concat_instance_indices = somen.file_io.load_array(working_dir / "instance_indices_valid.h5")
                sizes = somen.file_io.load_array(working_dir / "sizes_valid.h5")

            sections = np.cumsum(sizes[:-1])
            predictions = np.asarray(np.split(concat_predictions, sections), dtype=object)
            instance_indices = np.asarray(np.split(concat_instance_indices, sections), dtype=object)

            data.append((predictions, instance_indices))

        data = [np.concatenate([e[j] for e in data], axis=0) for j in range(2)]
        permutation = np.concatenate([valid_indices for _, valid_indices in folds], axis=0)
        inv_permutation = np.argsort(permutation)
        data = [e[inv_permutation] for e in data]

        concat_predictions = np.concatenate(data[0], axis=0)
        concat_instance_indices = np.concatenate(data[1], axis=0)
        sizes = np.asarray([len(e) for e in data[0]])

        out_dir = Path(f"data/working/cell_crop/{Path(config_path).stem}/")

        if public_hpa:
            somen.file_io.save_array(concat_predictions, out_dir / "predictions_valid_pub_hpa.h5")
            somen.file_io.save_array(concat_instance_indices, out_dir / "instance_indices_valid_pub_hpa.h5")
            somen.file_io.save_array(sizes, out_dir / "sizes_valid_pub_hpa.h5")
        else:
            somen.file_io.save_array(concat_predictions, out_dir / "predictions_valid.h5")
            somen.file_io.save_array(concat_instance_indices, out_dir / "instance_indices_valid.h5")
            somen.file_io.save_array(sizes, out_dir / "sizes_valid.h5")

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
