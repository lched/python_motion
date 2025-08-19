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


def smpl_to_bvh_data(smpl_dict, gender="NEUTRAL", frametime=1 / 60):
    rest_pose = np.array(SMPL_OFFSETS)

    root_offset = rest_pose[0]
    offsets = rest_pose  # - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100

    if "smpl_scaling" in smpl_dict.keys():
        scaling = smpl_dict["smpl_scaling"]
    else:
        scaling = 100

    rots = smpl_dict["smpl_poses"]
    rots = rots.reshape(rots.shape[0], -1, 3)  # (N, 24, 3)
    trans = smpl_dict["smpl_trans"]  # (N, 3)
    trans /= scaling

    # to quaternion
    rots = quat.from_axis_angle(rots)
    # order = "yzx"
    order = "zyx"

    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:, 0] += trans * 100
    rotations = quat.to_euler(rots, order=order)
    rotations = np.unwrap(rotations, axis=0)  # Unwrap in radians
    rotations = np.degrees(rotations)  # Convert back to degrees

    bvh_data = {
        "rotations": rotations,
        "positions": positions / 100,  # We want the results in meter convention
        "offsets": offsets / 100,
        "parents": SMPL_PARENTS,
        "names": SMPL_JOINTS_NAMES,
        "order": order,
        "frametime": frametime,
    }
    return bvh_data


def bvh_data_to_smpl(bvh_data):
    # First, make sure the bvh_data is in the same order as SMPL format expects
    # Create a mapping from the current names to the SMPL_JOINTS_NAMES
    name_to_index = {name: i for i, name in enumerate(bvh_data["names"])}

    # Create a reordering index array
    reorder_index = [name_to_index[name] for name in SMPL_JOINTS_NAMES]

    # Extract BVH data
    rotations = bvh_data["rotations"][:, reorder_index, :]
    positions = bvh_data["positions"][:, reorder_index, :]

    # Convert rotations
    rotations = np.radians(rotations)
    rotations = quat.from_euler(rotations, order=bvh_data["order"])
    rotations = quat.to_axis_angle(rotations)

    # Reshape rotations to match SMPL format
    rotations = rotations.reshape(rotations.shape[0], -1)

    # Extract root translation
    trans = positions[:, 0]

    # Prepare SMPL dictionary
    smpl_dict = {
        "smpl_poses": rotations,
        "smpl_trans": trans * 100,
        "smpl_scaling": np.array([100]),
    }

    return smpl_dict
