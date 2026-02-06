#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_to_matrix(position, quaternion):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quaternion).as_matrix()
    T[:3, 3] = position
    return T


def matrix_to_pose(T):
    position = T[:3, 3]
    quaternion = R.from_matrix(T[:3, :3]).as_quat()
    return position, quaternion


def cam_in_tooltip_frame(cam_pos_ee, cam_quat_ee, ee_tooltip_pos, ee_tooltip_quat):
    T_cam2ee = pose_to_matrix(cam_pos_ee, cam_quat_ee)
    T_tooltip2ee = pose_to_matrix(ee_tooltip_pos, ee_tooltip_quat)

    # ✔ Correct: express camera pose in tooltip frame
    T_cam2tooltip = np.linalg.inv(T_tooltip2ee) @ T_cam2ee

    return matrix_to_pose(T_cam2tooltip)


if __name__ == "__main__":
    cam_pos_ee = np.array([-0.07102005306238238,
                           0.024130938417833106,
                           0.022722818567460196])

    cam_quat_ee = np.array([-0.0050329509506691905,
                            0.0118942743151208,
                            -0.6630459166758809,
                            0.7484673059143502])

    ee_tooltip_pos = np.array([0.0, 0.0, 0.12])
    ee_tooltip_quat = np.array([0.0, 0.0, 0.0, 1.0])

    cam_pos_tooltip, cam_quat_tooltip = cam_in_tooltip_frame(
        cam_pos_ee, cam_quat_ee, ee_tooltip_pos, ee_tooltip_quat
    )

    print("Camera position in tooltip frame:", cam_pos_tooltip)
    print("Camera orientation in tooltip frame:", cam_quat_tooltip)
