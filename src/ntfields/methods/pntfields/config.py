import math
from dataclasses import dataclass, field

from ntfields.base import MethodConfig
from ntfields.data import NearSurface
from ntfields.methods.pntfields.schedule import AlphaSchedule


@dataclass
class PNTFieldsConfig(MethodConfig):
    """Defaults reproduce ruiqini/P-NTFields' `models/model_res_sigmoid.py` with its `test`/`c3d` data settings
    (margin 0.1, offset 0.01)."""

    lr: float = 1e-3
    weight_decay: float = 0.1
    # model
    hidden: int = 128
    encoder_blocks: int = 2
    generator_blocks: int = 3
    fourier_features: int = 128
    fourier_scale: float = 0.5
    beta: float = 10.0  # softplus β and the logsumexp scale of the smoothed max/min
    # loss
    viscosity: float = 1e-3  # γ in S_visc = 1/(1/S + γ·Δτ)
    alpha: AlphaSchedule = field(default_factory=AlphaSchedule)
    # training recipe
    lr_min: float = 5e-4
    lr_ramp_start: int = 8000  # lr falls linearly from `lr` after this epoch, reaching lr_min at start + length/2
    lr_ramp_length: int = 1000
    rollback_threshold: float | None = 1.2  # None: no rollback
    loss_scale_numerator: float = 50.0  # β = numerator / Σ batch losses: len(loader) of the reference's 5e5 pairs
    sampler: NearSurface = field(default_factory=lambda: NearSurface(margin=0.1, min_clearance=0.01, max_step=None))

    @classmethod
    def arm(cls) -> "PNTFieldsConfig":
        """The reference arm recipe: margin 0.5/12, goal step up to √3, 2e6 pairs (200 batches), and Fourier scale
        1.0, as in the published UR5 model (its B has std 1.0; the repo code says 0.5)."""
        margin = 0.5 / 12
        return cls(
            fourier_scale=1.0,
            loss_scale_numerator=200.0,
            sampler=NearSurface(margin=margin, min_clearance=margin / 10, max_step=math.sqrt(3)),
        )

    @classmethod
    def planar(cls) -> "PNTFieldsConfig":
        """The generic recipe with the α and lr schedules compressed 10x, for 1000-epoch point-mass runs."""
        return cls(alpha=AlphaSchedule(delay=200, period=400), lr_ramp_start=800, lr_ramp_length=100)
