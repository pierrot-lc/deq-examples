from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import jax.random as jr
from hydra.utils import to_absolute_path
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig


@dataclass
class DatasetConfig:
    max_samples: int | None
    n_classes: int
    test_path: Path
    train_path: Path

    def __post_init__(self):
        train_path = to_absolute_path(str(self.train_path))
        self.train_path = Path(train_path)

        test_path = to_absolute_path(str(self.test_path))
        self.test_path = Path(test_path)


@dataclass
class ImplicitConfig:
    solve_method: str
    tangent_solve_method: str


@dataclass
class ModelConfig:
    n_channels: int
    kernel_size: int
    key: PRNGKeyArray

    def __post_init__(self):
        self.key = jr.key(self.key)


@dataclass
class OptimizerConfig:
    learning_rate: float
    name: str


@dataclass
class TrainerConfig:
    batch_size: int
    eval_freq: int
    eval_iters: int
    gamma: float
    key: PRNGKeyArray
    total_iters: int

    def __post_init__(self):
        self.key = jr.key(self.key)


@dataclass
class WandBConfig:
    entity: str
    group: str
    mode: Literal["online", "offline", "disabled"]


@dataclass
class MainConfig:
    dataset: DatasetConfig
    implicit: ImplicitConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    wandb: WandBConfig

    @classmethod
    def from_dict(cls, config: DictConfig) -> Self:
        return cls(
            dataset=DatasetConfig(**config.dataset),
            implicit=ImplicitConfig(**config.implicit),
            model=ModelConfig(**config.model),
            optimizer=OptimizerConfig(**config.optimizer),
            trainer=TrainerConfig(**config.trainer),
            wandb=WandBConfig(**config.wandb),
        )
