from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration for experiment.
    """

    image_size: int = 128
    image_channels: int = 3
    n_classes: int = 10
    dataset: str = "cifar10"
    model: str = "MobileNetV2"

    batch_size: int = 64
    epochs: int = 10
    dropout: float = 0.4
    init_lr: float = 0.0005
    decay: float = 0.96
    trainable: bool = False


cfg = Config()