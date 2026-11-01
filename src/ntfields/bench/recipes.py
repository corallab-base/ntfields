import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Recipe:
    """How an entry trains: its reference recipe, so no method is held to another's settings."""

    epochs: int = 1000
    n_pairs: int = 10_000
    batch_size: int = 512
    batches_per_epoch: int | None = None
    micro_batch_size: int | None = None


RECIPES: dict[str, Recipe] = {
    "euclidean": Recipe(epochs=100),
    "td_ntfields_2d": Recipe(epochs=1000, n_pairs=1_000_000, batch_size=2000, batches_per_epoch=5),
    "pntfields_2d": Recipe(epochs=1000, n_pairs=500_000, batch_size=10_000, batches_per_epoch=6),
}


def register_recipe(preset: str, recipe: Recipe) -> None:
    RECIPES[preset] = recipe


def get_recipe(preset: str) -> Recipe:
    if preset not in RECIPES:
        log.warning("no recipe registered for %r; training with Recipe() defaults", preset)
    return RECIPES.get(preset, Recipe())
