"""This file contains methods related to the EDGE model https://github.com/Stanford-TML/EDGE
In its current implementation (21/05/25) EDGE takes as inputs motion data in the SMPL Y-up format (which we refer to as smpl) but outputs Z-up SMPL data (which we refer to as edge).
"""

import copy

import numpy as np
from scipy.spatial.transform import Rotation as R


def edge_to_smpl(edge_dict):
    """This function creates a copy of the dict!"""
    smpl_dict = copy.deepcopy(edge_dict)
    rotation = R.from_euler(
        "xyz", np.array([-90, 0, 0]), degrees=True
    )  # -90 degrees about the x axis
    root_rotvec = smpl_dict["smpl_poses"][:, 0]
    root_rotvec = (rotation * R.from_rotvec(root_rotvec)).as_rotvec()
    smpl_dict["smpl_poses"][:, 0] = root_rotvec

    # For the positions, swap Y and Z
    smpl_trans_y_up = np.copy(smpl_dict["smpl_trans"])
    smpl_trans_y_up[..., 1] = smpl_dict["smpl_trans"][..., 2]
    smpl_trans_y_up[..., 2] = -smpl_dict["smpl_trans"][
        ..., 1
    ]  # don't forget the minus!!!
    smpl_dict["smpl_trans"] = smpl_trans_y_up
    return smpl_dict


def smpl_to_edge(smpl_dict):
    edge_dict = copy.deepcopy(smpl_dict)
    rotation = R.from_euler(
        "xyz", np.array([90, 0, 0]), degrees=True
    )  # +90 degrees about the x axis
    root_rotvec = edge_dict["smpl_poses"][:, 0]
    root_rotvec = (rotation * R.from_rotvec(root_rotvec)).as_rotvec()
    edge_dict["smpl_poses"][:, 0] = root_rotvec

    # For the positions, swap Y and Z
    edge_trans_z_up = np.copy(edge_dict["smpl_trans"])
    edge_trans_z_up[..., 1] = -edge_dict["smpl_trans"][..., 2]
    edge_trans_z_up[..., 2] = edge_dict["smpl_trans"][
        ..., 1
    ]  # don't forget the minus!!!
    edge_dict["smpl_trans"] = edge_trans_z_up
    return edge_dict
