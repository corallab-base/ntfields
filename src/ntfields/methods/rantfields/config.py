from dataclasses import dataclass, field

from ntfields.base import MethodConfig
from ntfields.data import SurfaceMix, Uniform
from ntfields.methods.rantfields.sampler import JointSpaceMix


@dataclass
class RaNTFieldsConfig(MethodConfig):
    """Defaults reproduce the paper's 2D runs (point mass in [-1, 1]² under a flow or on a terrain)."""

    lr: float = 3e-4  # AdamW's base lr; OneCycle starts at max_lr / 25 regardless
    weight_decay: float = 1e-3
    # model
    embed: int = 32
    hidden: int = 256
    n_residual_blocks: int = 3
    n_finsler_blocks: int = 3
    # loss
    flow_scale: float = 1.0  # κ: the PDE sees κ·f; the arm recipe uses 0.8
    w_boundary: float = 5.0
    # training recipe
    max_lr: float = 1e-3  # OneCycleLR, stepped every batch
    pct_start: float = 0.1
    schedule_epochs: int | None = None  # epochs the LR cycle spans; None: the training budget's
    rollback_threshold: float | None = None  # None: no rollback
    sampler: SurfaceMix | Uniform | JointSpaceMix = field(
        # the reference: surface ends at clearance U(0, 0.1), always the goal (its T = 0 anchor)
        default_factory=lambda: SurfaceMix(surface_fraction=0.5, shell=0.1, perturb=0.5, anchor_end="goal")
    )

    @classmethod
    def arm(cls) -> "RaNTFieldsConfig":
        """The paper's 5-joint UR5e recipe (run 2026-09-15/02-52-19), whose reported checkpoint is epoch 100 of a
        400-epoch cycle: train with `Epochs(100)`."""
        return cls(
            hidden=512,
            n_residual_blocks=4,
            n_finsler_blocks=3,
            flow_scale=0.8,
            max_lr=3e-3,
            schedule_epochs=400,
            rollback_threshold=1.2,
            sampler=JointSpaceMix(),
        )
