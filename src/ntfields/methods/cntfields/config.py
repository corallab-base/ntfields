from dataclasses import dataclass, field

from ntfields.base import MethodConfig
from ntfields.methods.cntfields.schedule import LinearAlpha


@dataclass
class CNTFieldsConfig(MethodConfig):
    """Defaults reproduce ntfields-old's `cntfields_hold_cup.yaml`, `CNTFieldNetwork` and `NTFields.train`."""

    lr: float = 1e-3
    weight_decay: float = 0.01  # AdamW's default, which the reference keeps
    # model
    hidden: int = 256
    encoder_blocks: int = 2
    generator_blocks: int = 3
    fourier_features: int = 128
    fourier_scale: float = 1.0
    fourier_truncate: float | None = 2.0
    beta: float = 10.0  # softplus β and the logsumexp scale of the smoothed max/min
    # loss
    viscosity: float = 2e-3  # γ in S_visc = 1/(1/S + γ·Δτ)
    alpha: LinearAlpha = field(default_factory=LinearAlpha)
    # training recipe
    rollback_threshold: float | None = 1.2  # None: no rollback

    @classmethod
    def compressed(cls) -> "CNTFieldsConfig":
        """The α schedule compressed 5x for 1000-epoch runs on low-dimensional manifolds."""
        return cls(alpha=LinearAlpha(step=5e-4))
