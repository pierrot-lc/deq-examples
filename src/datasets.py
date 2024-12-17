from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray
from PIL import Image
from tqdm import tqdm


@dataclass
class MNISTDataset:
    images: list[Int[Array, "height width"]]
    classes: list[Int[Array, ""]]

    def __post_init__(self):
        assert len(self.images) == len(self.classes)

    def __len__(self) -> int:
        return len(self.images)

    def dataloader(
        self, batch_size: int, n_iterations: int, *, key: PRNGKeyArray
    ) -> Iterator[
        tuple[Float[Array, "batch_size height width"], Int[Array, " batch_size"]]
    ]:
        for sk in jr.split(key, n_iterations):
            batch_ids = jr.randint(sk, shape=(batch_size,), minval=0, maxval=len(self))
            x = [self.images[id_] for id_ in batch_ids]
            y = [self.classes[id_] for id_ in batch_ids]

            x = jnp.stack(x)
            y = jnp.stack(y)

            x = x / 255
            yield x, y

    @classmethod
    def from_directory(cls, directory: Path, max_samples: int | None = None) -> Self:
        classes_dir = [d for d in directory.glob("*") if d.is_dir()]

        paths = [
            (int(dir_.stem), path_)
            for dir_ in classes_dir
            for path_ in dir_.glob("*.png")
        ]

        if max_samples is not None:
            # NOTE: The samples are not shuffled before this cutoff. Classes won't be
            # uniformly represented. This argument should be used for debug purposes
            # only.
            paths = paths[:max_samples]

        images = []
        classes = []
        for digit_, path_ in tqdm(paths, "Reading images", leave=False):
            classes.append(jnp.array(digit_))
            with Image.open(path_) as img:
                img = jnp.asarray(img)
                images.append(img)

        return cls(images, classes)
