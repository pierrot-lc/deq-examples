import hydra
import wandb
from configs import MainConfig
from omegaconf import DictConfig, OmegaConf
from optax import adamw
from src.datasets import MNISTDataset
from src.solvers import Solver
from src.model import ConvNet
from src.trainer import Trainer


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(dict_config: DictConfig):
    config = MainConfig.from_dict(dict_config)

    train_dataset = MNISTDataset.from_directory(
        config.dataset.train_path, config.dataset.max_samples
    )
    test_dataset = MNISTDataset.from_directory(
        config.dataset.test_path, config.dataset.max_samples
    )

    model = ConvNet(
        config.model.n_channels,
        config.model.kernel_size,
        config.dataset.n_classes,
        key=config.model.key,
    )

    match config.optimizer.name:
        case "adamw":
            optimizer = adamw(config.optimizer.learning_rate)
        case _:
            raise ValueError(f"Unknown optimizer: {config.optimizer.name}")

    solver = Solver(config.implicit.n_iterations, config.implicit.anderson_m)

    trainer = Trainer(
        config.trainer.batch_size,
        config.trainer.eval_freq,
        config.trainer.eval_iters,
        config.trainer.gamma,
        optimizer,
        solver,
        config.trainer.total_iters,
    )

    with wandb.init(
        project="deq-examples",
        group=config.wandb.group,
        config=OmegaConf.to_container(dict_config),
        entity=config.wandb.entity,
        mode=config.wandb.mode,
    ) as run:
        trainer.train(model, train_dataset, test_dataset, run, key=config.trainer.key)


if __name__ == "__main__":
    main()
