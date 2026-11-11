# pinocchio ships no type stubs
# pyright: reportAttributeAccessIssue=false

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Dynamics(Protocol):
    """Rigid-body dynamics of the actuated joints, in physical units: M(q) q̈ + C(q, q̇) q̇ + g(q) = τ."""

    def mass_matrix(self, q: np.ndarray) -> np.ndarray: ...
    def inverse_dynamics(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray) -> np.ndarray: ...


@dataclass
class PointMassDynamics:
    """A point mass driven by direct forces: τ = m q̈."""

    mass: float = 1.0

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return self.mass * np.eye(len(q))

    def inverse_dynamics(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray) -> np.ndarray:
        return self.mass * np.asarray(qddot)


class PinocchioDynamics:
    """Dynamics of `active_joints` of a URDF robot, with every other joint locked at 0 (needs the `[dynamics]`
    extra). `robot` is "ur5e" (the bundled UR5e) or a path to a URDF."""

    def __init__(self, robot: str, active_joints: list[str]) -> None:
        import pinocchio as pin

        self._pin = pin
        full = load_pinocchio_model(robot)
        names = [full.names[i] for i in range(1, full.njoints)]
        locked = [full.getJointId(n) for n in names if n not in active_joints]
        self.model = pin.buildReducedModel(full, locked, np.zeros(full.nq))
        self.data = self.model.createData()

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        upper = np.triu(self._pin.crba(self.model, self.data, q))  # crba fills only the upper triangle
        return upper + upper.T - np.diag(np.diag(upper))

    def gravity(self, q: np.ndarray) -> np.ndarray:
        return self._pin.computeGeneralizedGravity(self.model, self.data, q)

    def inverse_dynamics(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray) -> np.ndarray:
        return self._pin.rnea(self.model, self.data, q, qdot, qddot)


def load_pinocchio_model(robot: str):  # noqa: ANN201  (pinocchio has no type stubs)
    """A Pinocchio model from a URDF path, or "ur5e" for the bundled UR5e (`assets/ur5e_spherized/ur5e.urdf`)."""
    import pinocchio as pin

    if robot == "ur5e":
        from ntfields.envs.arm import UR5E_SPHERIZED

        robot = str(UR5E_SPHERIZED / "ur5e.urdf")
    return pin.buildModelFromUrdf(robot)
