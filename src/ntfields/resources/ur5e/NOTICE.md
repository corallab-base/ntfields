# UR5e assets

Copied from ntrl-demo (github.com/ruiqini/ntrl-demo, commit `7a4aaf2`, `datasets/arm/UR5/`), the TD-NTFields
reference code. ntrl-demo ships no license file.

- `ur5e.urdf`: the UR5e description used by ntrl-demo. Only the kinematic chain is read; the mesh files it names
  are not shipped.
- `spheres/<link>.npy`: ntrl-demo's collision spheres, rows (x, y, z, r) in each link's frame.
- `ntrl_scene_points.npy`: the vertices of ntrl-demo's scanned lab scene (`realpc_scaled.off`), in the robot base
  frame (m).
