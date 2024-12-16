from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, jaxtyped
from tqdm import tqdm
from wandb.data_types import WBValue
from wandb.wandb_run import Run

from .datasets import MNISTDataset
from .model import ConvNet


class Trainer(eqx.Module):
    batch_size: int
    eval_freq: int
    eval_iters: int
    optimizer: optax.GradientTransformation
    total_iters: int

    def train(
        self,
        model: ConvNet,
        train_dataset: MNISTDataset,
        test_dataset: MNISTDataset,
        logger: Run,
        *,
        key: PRNGKeyArray,
    ):
        n_params = Trainer.count_params(model)
        logger.summary["params"] = n_params
        print(f"{n_params:,} params")

        logger.summary["training-size"] = len(train_dataset)
        logger.summary["validation-size"] = len(test_dataset)

        logger.define_metric("train.loss", summary="min")
        logger.define_metric("val.loss", summary="min")

        params, _ = eqx.partition(model, eqx.is_array)
        opt_state = self.optimizer.init(params)
        keys = Trainer.keys_generator(key)

        for iter_id, (x, y) in tqdm(
            enumerate(
                train_dataset.dataloader(
                    self.batch_size, self.total_iters, key=next(keys)
                )
            ),
            desc="Training",
            total=self.total_iters,
        ):
            if iter_id % self.eval_freq == 0:
                metrics = {
                    "train": self.eval(model, train_dataset, key=next(keys)),
                    "test": self.eval(model, test_dataset, key=next(keys)),
                }
                logger.log(metrics, step=iter_id)

            model, opt_state = self.batch_update(model, x, y, opt_state)
            assert Trainer.is_finite(model), "Non-finite weights!"

        # Final evals.
        metrics = {
            "train": self.eval(model, train_dataset, key=next(keys)),
            "test": self.eval(model, test_dataset, key=next(keys)),
        }
        logger.log(metrics, step=iter_id)

    def eval(
        self, model: ConvNet, dataset: MNISTDataset, *, key: PRNGKeyArray
    ) -> dict[str, WBValue]:
        keys = Trainer.keys_generator(key)

        metrics = [
            self.batch_metrics(model, x, y)
            for x, y in tqdm(
                dataset.dataloader(self.batch_size, self.eval_iters, key=next(keys)),
                desc="Eval",
                total=self.eval_iters,
                leave=False,
            )
        ]
        metrics = jax.tree.map(lambda *xs: jnp.concat(xs), *metrics)
        metrics = jax.tree.map(jnp.mean, metrics)
        metrics = jax.tree.map(float, metrics)
        return metrics

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def batch_update(
        self,
        model: ConvNet,
        x: Float[Array, "batch_size height width"],
        y: Int[Array, " batch_size"],
        opt_state: optax.OptState,
    ) -> tuple[ConvNet, optax.OptState]:
        def batch_loss(model: ConvNet) -> Scalar:
            y_hat = eqx.filter_vmap(model)(x)
            losses = optax.losses.softmax_cross_entropy_with_integer_labels(y_hat, y)
            return losses.mean()

        grads = eqx.filter_grad(batch_loss)(model)
        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)
    def batch_metrics(
        self,
        model: ConvNet,
        x: Float[Array, "batch_size height width"],
        y: Int[Array, " batch_size"],
    ) -> dict[str, Float[Array, " batch_size"]]:
        y_hat = eqx.filter_vmap(model)(x)
        return {
            "loss": optax.losses.softmax_cross_entropy_with_integer_labels(y_hat, y),
            "acc": (y_hat.argmax(axis=1) == y).astype(float),
        }

    @staticmethod
    def count_params(model: eqx.Module) -> Int[Array, ""]:
        """Count the number of parameters of the given equinox module."""
        # Replace the params of the PE module by None to filter them out.
        model = jax.tree_util.tree_map_with_path(
            lambda p, v: None
            if "positional_encoding" in jax.tree_util.keystr(p)
            else v,
            model,
        )
        # jax.tree_util.tree_map_with_path(lambda p, _: print(p), model)

        params = eqx.filter(model, eqx.is_array)
        n_params = jax.tree.map(lambda p: jnp.prod(jnp.array(p.shape)), params)
        n_params = jnp.array(jax.tree.leaves(n_params))
        n_params = jnp.sum(n_params)
        return n_params

    @eqx.filter_jit
    @staticmethod
    def is_finite(model: eqx.Module) -> Scalar:
        """True if all parameters of the model are finite, False otherwise."""
        params = eqx.filter(model, eqx.is_array)
        are_finite = jax.tree.map(lambda p: jnp.all(jnp.isfinite(p)), params)
        # NOTE: The proper way would be to use `jax.tree.all` but it is not jittable.
        return jax.tree.reduce(lambda x, y: x & y, are_finite)

    @staticmethod
    def keys_generator(key: PRNGKeyArray) -> Iterator[PRNGKeyArray]:
        """Generate infinitely many keys on demand."""
        while True:
            key, sk = jr.split(key)
            yield sk
