# UR5e + Robotiq 2F-85 collision spheres

`spheres/<link>.npy` holds rows (x, y, z, r) in each link's frame, read from `ur5e_robotiq_2f85_spherized.urdf` in the
WoS-CBF project. The gripper is rigid and fully open, and its 7 spheres are on `wrist_3_link`. The static base sphere
is dropped: the pedestal in each scene covers it.

The kinematics are those of `../ur5e/ur5e.urdf` (forward kinematics agree to numerical precision), so that URDF is used.

`self_ignore.json` lists the link pairs that `ArmEnv.self_distance` leaves out, from 200k uniform configurations: the
five adjacent pairs overlap in every configuration, and the two others never collide but stay within 1.1 cm.

`visual.npz` holds one visual mesh per arm link, in the link's frame, for the viewer; the gripper is merged into
`wrist_3_link`. The UR5e meshes are from the `ur_description` package, under its BSD 3-Clause licence. The 2F-85 meshes come from the WoS-CBF project; their origin is not recorded there, so check it
before redistributing them.
