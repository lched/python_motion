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
                scaling = (1,)
                print("WARNING: No scaling found in the file, defaults to 1.")
            trans = smpl_file["smpl_trans"]  # (N, 3)
    else:
        raise ValueError("This file type is not supported!")
    smpl_dict = {"smpl_poses": rots, "smpl_trans": trans, "smpl_scaling": scaling}
    return smpl_dict


def smpl_to_bvh_data(smpl_dict, gender="NEUTRAL", frametime=1 / 60, scale=1):
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
    offsets *= scale

    rots = smpl_dict["smpl_poses"]
    rots = rots.reshape(rots.shape[0], -1, 3)  # (N, 24, 3)
    positions = np.zeros_like(rots)
    positions[:, 0] = smpl_dict["smpl_trans"]  # (N, 3)

    # to quaternion
    rots = quat.from_axis_angle(rots)
    # print("quats shape:", quat.shape)

    order = "yzx"
    rotations = np.degrees(quat.to_euler(rots, order=order))

    bvh_data = {
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": SMPL_JOINTS_NAMES,
        "order": order,
        "frametime": frametime,
    }
    return bvh_data


def bvh_data_to_smpl(bvh_data, gender="NEUTRAL"):
    # Extract BVH data
    rotations = bvh_data["rotations"]
    positions = bvh_data["positions"]
    offsets = bvh_data["offsets"]
    order = "zyx"

    # Convert rotations from degrees to radians
    rotations = np.radians(rotations)  # TODO: check that

    # Convert rotations from Euler angles to quaternions
    rots = quat.from_euler(rotations, order=order)

    # Convert quaternions to axis-angle representation
    rots = quat.to_scaled_angle_axis(rots)

    # Reshape rotations to match SMPL format
    rots = rots.reshape(rots.shape[0], -1)

    # Extract root translation and scale it back
    trans = positions[:, 0] - offsets[0][None]
    trans /= 100

    # Create SMPL model to get scaling factor
    model = smplx.create(
        model_path=Path(__file__).parent / "smpl_utils/data/smpl/",
        model_type="smpl",
        gender=gender,
        batch_size=1,
    )
    rest = model()
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24, :]
    root_offset = rest_pose[0]
    scaling = np.linalg.norm(root_offset) / np.linalg.norm(offsets[0])

    # Scale translation back
    trans *= scaling

    # Prepare SMPL dictionary
    smpl_dict = {
        "smpl_poses": rots,
        "smpl_trans": trans,
        "smpl_scaling": np.array([scaling]),
    }

    return smpl_dict
