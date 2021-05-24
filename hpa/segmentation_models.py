from dataclasses import dataclass

import segmentation_models_pytorch as smp
from torch import nn


@dataclass
class ModelConfig:
    arch: str


def build_model(config: ModelConfig, pretrained: bool = True, out_channels: int = 19) -> nn.Module:
    if config.arch.startswith("unet_"):
        encoder_name = config.arch[len("unet_") :]
        encoder_weights = "imagenet" if pretrained else None
        return smp.Unet(encoder_name=encoder_name, in_channels=4, classes=out_channels, encoder_weights=encoder_weights)

    raise ValueError
