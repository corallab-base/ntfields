"""The MuJoCo Menagerie UR5e driven by joint torques, with joints outside the active set locked at 0 (as in the
reference's closed-loop runs). Needs the `sim` dependency group."""

# mujoco ships no type stubs
# pyright: reportAttributeAccessIssue=false
import copy

import mujoco
import numpy as np

UR5E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def torque_model(active: tuple[str, ...], contacts: bool = False) -> mujoco.MjModel:
    """Menagerie UR5e with its PD servos turned into torque motors (ctrlrange = forcerange) and the other joints held
    at 0 by joint equality constraints. Contacts are off by default, as in the paper's runs: collisions are judged
    against the planning scene afterwards (`run.summarize`), not by MuJoCo."""
    from robot_descriptions import ur5e_mj_description

    spec = mujoco.MjSpec.from_file(ur5e_mj_description.MJCF_PATH)
    for actuator in spec.actuators:
        force_range = list(actuator.forcerange)
        actuator.set_to_motor()
        actuator.ctrlrange = force_range
        actuator.ctrllimited = True
    for name in UR5E_JOINTS:
        if name not in active:
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
            eq.name1 = name
    model = spec.compile()
    if not contacts:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return model


def addresses(model: mujoco.MjModel, active: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """qpos addresses, dof addresses, actuator ids and torque limits of the active joints."""
    qpos = np.array([model.joint(n).qposadr[0] for n in active])
    dof = np.array([model.joint(n).dofadr[0] for n in active])
    act = []
    for n in active:
        jid = model.joint(n).id
        (match,) = [i for i in range(model.nu) if model.actuator_trnid[i, 0] == jid]
        act.append(match)
    act = np.array(act)
    return qpos, dof, act, model.actuator_forcerange[act, 1].copy()


class MuJoCoDynamics:
    """Inverse dynamics of the active joints on (a contact-free copy of) the simulated model itself, so the
    controller's feedforward matches the plant exactly."""

    def __init__(self, model: mujoco.MjModel, active: tuple[str, ...]) -> None:
        self.model = copy.deepcopy(model)
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.data = mujoco.MjData(self.model)
        self.qpos, self.dof, _, _ = addresses(model, active)

    def _set(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qpos[self.qpos] = q
        self.data.qvel[self.dof] = qdot
        self.data.qacc[self.dof] = qddot

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        self._set(q, np.zeros_like(q), np.zeros_like(q))
        mujoco.mj_forward(self.model, self.data)
        full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, full)
        return full[np.ix_(self.dof, self.dof)]

    def inverse_dynamics(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray) -> np.ndarray:
        self._set(q, qdot, qddot)
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[self.dof].copy()
