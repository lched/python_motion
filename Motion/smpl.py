# From https://github.com/KosukeFukazawa/smpl2bvh/blob/main/smpl2bvh.py
from pathlib import Path
import pickle

import numpy as np
import smplx

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
    model = smplx.create(
        model_path=Path(__file__).parent / "smpl_utils/data/smpl/",
        model_type="smpl",
        gender=gender,
        batch_size=1,
    )
    parents = model.parents.detach().cpu().numpy()
    rest = model()
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24, :]

    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
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
    order = "yzx"

    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:, 0] += trans * 100
    rotations = np.degrees(quat.to_euler(rots, order=order))

    bvh_data = {
        "rotations": rotations,
        "positions": positions / 100,  # We want the results in meter convention
        "offsets": offsets / 100,
        "parents": parents,
        "names": SMPL_JOINTS_NAMES,
        "order": order,
        "frametime": frametime,
    }
    return bvh_data


def bvh_data_to_smpl(bvh_data, gender="NEUTRAL"):
    # First, make sure the bvh_data is in the same order as SMPL format expects
    # Create a mapping from the current names to the SMPL_JOINTS_NAMES
    name_to_index = {name: i for i, name in enumerate(bvh_data["names"])}
    # smpl_to_index = {name: i for i, name in enumerate(SMPL_JOINTS_NAMES)}
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

    # Extract root translation and scale it back
    trans = positions[:, 0]  # - offsets[0][None]

    # Prepare SMPL dictionary
    smpl_dict = {
        "smpl_poses": rotations,
        "smpl_trans": trans,
        "smpl_scaling": np.array([100]),
    }

    return smpl_dict
