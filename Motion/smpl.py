# From https://github.com/KosukeFukazawa/smpl2bvh/blob/main/smpl2bvh.py
import pickle

import numpy as np

# import torch

# from .Animation import Animation
# from .Quaternions import Quaternions
from .smpl_utils.utils import quat

# from .transforms import euler2mat, mat2quat

SMPL_JOINTS_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

SMPL_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]

SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

LISN_JOINTS_NAMES = [
    "Hips",  # 0
    "Spine",  # 1
    "Spine1",  # 2
    "Neck",  # 3
    "Head",  # 4
    "LeftShoulder",  # 5
    "LeftArm",  # 6
    "LeftForeArm",  # 7
    "LeftHand",  # 8
    "RightShoulder",  # 9
    "RightArm",  # 10
    "RightForeArm",  # 11
    "RightHand",  # 12
    "LeftUpLeg",  # 13
    "LeftLeg",  # 14
    "LeftFoot",  # 15
    "LeftToeBase",  # 16
    "RightUpLeg",  # 17
    "RightLeg",  # 18
    "RightFoot",  # 19
    "RightToeBase",  # 20
]

LISN_OFFSETS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.070852, 0.0],
    [0.0, 0.152636, 0.0],
    [0.0, 0.169916, 0.0],
    [0.0, 0.133927, 0.017281],
    [0.0, 0.17281, 0.0],
    [0.034198, 0.131853, -0.003205],
    [0.115879, 0.0, 0.0],
    [0.252722, 0.0, 0.0],
    [0.231372, 0.0, 0.0],
    [0.129607, 0.0, 0.0],
    [-0.03481, 0.131853, -0.003205],
    [-0.115879, 0.0, 0.0],
    [-0.252722, 0.0, 0.0],
    [-0.231372, 0.0, 0.0],
    [-0.129607, 0.0, 0.0],
    [0.086405, 0.0, 0.0],
    [0.0, -0.403038, 0.0],
    [0.0, -0.374775, 0.0],
    [0.0, -0.056163, 0.129607],
    [0.0, 0.0, 0.034562],
    [-0.086405, 0.0, 0.0],
    [0.0, -0.403038, 0.0],
    [0.0, -0.374775, 0.0],
    [0.0, -0.056163, 0.129607],
    [0.0, 0.0, 0.034562],
]

LISN_PARENTS = [
    -1,  # 0
    0,  # 1
    1,  # 2
    2,  # 3
    3,  # 4
    2,  # 5
    6,  # 6
    7,  # 7
    8,  # 8
    2,  # 9
    11,  # 10
    12,  # 11
    13,  # 12
    0,  # 13
    16,  # 14
    17,  # 15
    18,  # 16
    0,  # 17
    21,  # 18
    22,  # 19
    23,  # 20
]


def load_smpl(smpl_file):
    """Open animation in the SMPL format contained in a pickle or numpy data file.

    Args:
        smpl_file (str): Path to file

    Raises:
        ValueError: If the filename does not end with pkl or npz.

    Returns:
        smpl_dict: Dictionary with keys 'smpl_poses', 'smpl_trans' and 'smpl_scaling'
        as defined by the SMPL paper.
    """
    if smpl_file.endswith(".npz"):
        smpl_file = np.load(smpl_file)
        rots = np.squeeze(smpl_file["poses"], axis=0)  # (N, 24, 3)
        trans = np.squeeze(smpl_file["trans"], axis=0)  # (N, 3)

    elif smpl_file.endswith(".pkl"):
        with open(smpl_file, "rb") as f:
            smpl_file = pickle.load(f)
            rots = smpl_file["smpl_poses"]  # (N, 72)
            rots = rots.reshape(rots.shape[0], -1, 3)  # (N, 24, 3)
            if "smpl_scaling" in smpl_file.keys():
                scaling = smpl_file["smpl_scaling"]  # (1,)
            else:
                scaling = (100,)
                print("WARNING: No scaling found in the file, defaults to 100.")
            trans = smpl_file["smpl_trans"]  # (N, 3)
    else:
        raise ValueError("This file type is not supported!")
    smpl_dict = {"smpl_poses": rots, "smpl_trans": trans, "smpl_scaling": scaling}
    return smpl_dict


def smpl_to_bvh_data(smpl_dict, frametime=1 / 60, motion_format="smpl"):
    """ Convert SMPL dict (axis-angle poses + root trans) into BVH-style dict. \
        The motion channels should be in the order specified by FORMAT_JOINTS_NAMES \
        specified at the top of this file.

    Args:
        smpl_dict (dict): Dict with motion data as rotations in axis-angle and translations in meters.
        frametime (float, optional): In seconds. Defaults to 1/60.
        motion_format (['smpl', 'lisn'], optional): Motion format to use (defines joints names and order). Defaults to "smpl".

    Returns:
        bvh_data (dict): Motion data in the bvh format.
    """
    if motion_format == "smpl":
        parents = SMPL_PARENTS
        names = SMPL_JOINTS_NAMES
        rest_pose = np.array(SMPL_OFFSETS)
    elif motion_format == "lisn":
        parents = LISN_PARENTS
        names = LISN_JOINTS_NAMES
        rest_pose = np.array(LISN_OFFSETS)
    else:
        raise ValueError("motion_format must be 'lisn' or 'smpl'.")

    # Skeleton offsets
    offsets = rest_pose.copy()
    offsets[0] = rest_pose[0]  # root absolute
    offsets *= 100  # cm convention (BVH often uses cm)

    # Scaling
    scaling = smpl_dict.get("smpl_scaling", 100)

    rots = smpl_dict["smpl_poses"].reshape(-1, len(names), 3)  # (N, J, 3)
    trans = smpl_dict["smpl_trans"] / scaling  # (N, 3) in meters

    # Convert axis-angle → quaternion → Euler
    quats = quat.from_axis_angle(rots)
    order = "zyx"
    eulers = quat.to_euler(quats, order=order)  # radians
    eulers = np.unwrap(eulers, axis=0)  # temporal continuity
    eulers = np.degrees(eulers)  # BVH expects degrees

    # Root positions
    positions = np.tile(offsets[None], (len(rots), 1, 1))
    positions[:, 0] += trans * 100  # cm

    bvh_data = {
        "rotations": eulers,
        "positions": positions / 100,  # back to meters
        "offsets": offsets / 100,  # meters
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": frametime,
    }
    return bvh_data


def bvh_data_to_smpl(bvh_data, motion_format="smpl"):
    """Convert BVH dict back into SMPL-compatible dict.

    Args:
        bvh_data (dict): _description_
        motion_format (['smpl', 'lisn'], optional): Motion format to use (defines joints names and order). Defaults to "smpl".

    Returns:
        _type_: _description_
    """
    # First, make sure the bvh_data is in the same order as SMPL format expects
    # Create a mapping from the current names to the SMPL_JOINTS_NAMES
    name_to_index = {name: i for i, name in enumerate(bvh_data["names"])}
    if motion_format == "smpl":
        joints_names = SMPL_JOINTS_NAMES
    elif motion_format == "lisn":
        joints_names = LISN_JOINTS_NAMES
    else:
        raise ValueError("motion_format must be 'lisn' or 'smpl'.")
    reorder_index = [name_to_index[name] for name in joints_names]

    # Extract BVH data in SMPL order
    rotations = bvh_data["rotations"][:, reorder_index, :]  # degrees
    positions = bvh_data["positions"][:, reorder_index, :]  # meters

    # Convert Euler → quaternion → axis-angle
    order = bvh_data["order"]
    rotations = np.radians(rotations)
    quats = quat.from_euler(rotations, order=order)
    axis_angles = quat.to_axis_angle(quats)  # (N, J, 3)

    # Extract root translation (meters → cm for SMPL convention)
    trans = positions[:, 0] * 100

    # Extract root translation
    trans = positions[:, 0]

    smpl_dict = {
        "smpl_poses": smpl_poses,
        "smpl_trans": trans,
        "smpl_scaling": np.array([100]),  # cm convention
    }

    return smpl_dict
