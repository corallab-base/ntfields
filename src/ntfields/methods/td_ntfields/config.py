from dataclasses import dataclass, field

from ntfields.base import MethodConfig
from ntfields.data import NearSurface, SurfaceMix, UR5eSurfacePairs


@dataclass
class TDNTFieldsConfig(MethodConfig):
    """Defaults reproduce ntrl-demo's generic `models/metric` code. The arm presets use `models/metric_arm` at the
    width of ntrl's published UR5 run (256; the repo code says 512)."""

    lr: float = 5e-4  # the reference resets lr to 5e-4 every epoch
    weight_decay: float = 0.5
    # model
    hidden: int = 256
    n_blocks: int = 2
    group: int = 16
    temperature: float = 10.0
    out_scale: float = 0.2
    fourier_scale: float = 1.0
    # loss
    w_eikonal: float = 1e-2
    w_td: float = 1e-3
    w_normal: float = 1e-3
    causality: float = 0.5  # λ_C in exp(-λ_C·T)
    dt: float = 0.03  # TD step Δt
    detach_causality: bool = False  # True: exp(-λ_C·T) is a stop-gradient weight
    # training recipe
    rollback_threshold: float | None = 1.2  # None: no rollback
    loss_scale_numerator: float = 4.0  # β = numerator / Σ batch losses of the last epoch; ntrl-demo's Σ/4
    sampler: NearSurface | SurfaceMix | UR5eSurfacePairs = field(
        default_factory=lambda: NearSurface(margin=0.1, min_clearance=0.01, goal_min_clearance=-0.01, max_step=None)
    )

    @classmethod
    def arm(cls) -> "TDNTFieldsConfig":
        """The fixed UR5e recipe: IK surface pairs, and gradient clipping instead of rollback (ntrl-demo's
        recipe is only stable on its partly corrupted labels)."""
        return cls(
            hidden=256,
            out_scale=0.1,
            fourier_scale=0.2,
            dt=0.02,
            sampler=UR5eSurfacePairs(),
            grad_clip_norm=5.0,
            rollback_threshold=None,
        )

    @classmethod
    def arm_ntrl(cls) -> "TDNTFieldsConfig":
        """ntrl-demo's UR5 recipe: rollback and `NearSurface` pairs."""
        return cls(
            hidden=256, out_scale=0.1, fourier_scale=0.2, dt=0.02, sampler=NearSurface(margin=0.05, max_step=0.5)
        )

    @classmethod
    def planar(cls) -> "TDNTFieldsConfig":
        """The recipe for point masses in [-1, 1]^2: the arm model and surface-mixed pairs."""
        return cls(out_scale=0.1, fourier_scale=0.2, dt=0.02, sampler=SurfaceMix())

    @classmethod
    def stable(cls) -> "TDNTFieldsConfig":
        return cls(detach_causality=True)
