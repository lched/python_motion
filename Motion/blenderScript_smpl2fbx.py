import bpy
import os
import pickle
from pathlib import Path
import argparse
import glob
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R


class SmplObjects(object):
    joints = [
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

    def __init__(self, read_path):
        self.files = {}
        paths = sorted(glob.glob(os.path.join(read_path, "*.pkl")))
        for path in paths:
            filename = path.split("/")[-1]
            with open(path, "rb") as fp:
                data = pickle.load(fp)
            self.files[filename] = {
                "smpl_poses": data["smpl_poses"],
                "smpl_trans": data["smpl_trans"],
            }
        self.keys = list(self.files.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        return key, self.files[key]


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


# Calculate quaternions from axis angles.
def from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., None]
    s = np.sin(angle / 2.0)[..., None]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


# Calculate quaternions from axis-angle.
def from_axis_angle(rots):
    angle = np.linalg.norm(rots, axis=-1)
    # Prevent division by zero
    axis = np.where(
        angle[..., None] > 0,
        rots / angle[..., None],
        np.array([1.0, 0.0, 0.0]),  # default axis when angle == 0
    )
    return from_angle_axis(angle, axis)


def to_euler(x, order="zyx"):
    # Ensure the quaternions are normalized
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)

    q0 = x[..., 0:1]
    q1 = x[..., 1:2]
    q2 = x[..., 2:3]
    q3 = x[..., 3:4]

    if order == "zyx":
        return np.concatenate(
            [
                np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)),
                np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1, 1)),
                np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2)),
            ],
            axis=-1,
        )

    elif order == "yzx":
        return np.concatenate(
            [
                np.arctan2(
                    2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0
                ),
                np.arcsin((2 * (q1 * q2 + q3 * q0)).clip(-1, 1)),
                np.arctan2(
                    2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0
                ),
            ],
            axis=-1,
        )

    elif order == "zxy":
        return np.concatenate(
            [
                np.arctan2(
                    2 * (q0 * q3 - q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
                ),
                np.arcsin((2 * (q0 * q1 + q2 * q3)).clip(-1, 1)),
                np.arctan2(
                    2 * (q0 * q2 - q1 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
                ),
            ],
            axis=-1,
        )

    elif order == "yxz":
        return np.concatenate(
            [
                np.arctan2(
                    2 * (q1 * q3 + q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
                ),
                np.arcsin((2 * (q0 * q1 - q2 * q3)).clip(-1, 1)),
                np.arctan2(
                    2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
                ),
            ],
            axis=-1,
        )

    else:
        raise NotImplementedError("Cannot convert from ordering %s" % order)


def add_animation(
    pkl_name, fbx_source_path, smpl_params, output_folder, fps, fix_offset
):
    clear_scene()

    bpy.ops.import_scene.fbx(
        filepath=fbx_source_path,
        use_custom_props=True,
        use_custom_props_enum_as_string=True,
        ignore_leaf_bones=True,
    )
    bpy.context.scene.render.fps = fps

    armature = next(
        (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"), None
    )
    if not armature:
        raise Exception("No armature found in the scene.")
    armature.name = str(Path(pkl_name).stem)

    armature.animation_data_clear()
    if armature.animation_data is None:
        armature.animation_data_create()
    if armature.animation_data.action is None:
        armature.animation_data.action = bpy.data.actions.new("Action")

    # rotation = R.from_quat(np.array([-0.7071068, 0, 0, 0.7071068]))
    # root_rotvec = smpl_params["smpl_poses"][:, 0:3]
    # root_rotvec = (rotation * R.from_rotvec(root_rotvec)).as_rotvec()
    # smpl_params["smpl_poses"][:, 0:3] = root_rotvec

    # smpl_trans_y_up = np.copy(smpl_params["smpl_trans"])
    # smpl_trans_y_up[..., 1] = smpl_params["smpl_trans"][..., 2]
    # smpl_trans_y_up[..., 2] = -smpl_params["smpl_trans"][..., 1]
    # smpl_params["smpl_trans"] = smpl_trans_y_up
    # if fix_offset:
    #     smpl_params["smpl_trans"] -= [0, 0.9, 0]

    quats = from_axis_angle(
        smpl_params["smpl_poses"].reshape(smpl_params["smpl_poses"].shape[0], -1, 3)
    )

    joints = SmplObjects.joints
    for joint_idx, name in enumerate(joints):
        bone = armature.pose.bones.get(name)
        if not bone:
            print(f"{name} was not found!")
            continue
        bone.rotation_mode = "QUATERNION"

        for axis_idx, axis in enumerate(["w", "x", "y", "z"]):
            data_path = f'pose.bones["{name}"].rotation_quaternion'
            fcurve = armature.animation_data.action.fcurves.new(
                data_path, index=axis_idx
            )
            frames = np.arange(quats.shape[0])  # Frame indices
            samples = quats[:, joint_idx, axis_idx]  # Values for this axis
            # Add keyframes
            fcurve.keyframe_points.add(count=len(frames))
            fcurve.keyframe_points.foreach_set(
                "co", [x for co in zip(frames, samples) for x in co]
            )
            fcurve.update()

    # Translation
    bone = armature.pose.bones.get(joints[0])
    if bone:
        smpl_trans = smpl_params["smpl_trans"] / 100
        for axis_idx, axis in enumerate(["x", "y", "z"]):
            data_path = f'pose.bones["{joints[0]}"].location'
            fcurve = armature.animation_data.action.fcurves.new(
                data_path, index=axis_idx
            )
            frames = np.arange(smpl_trans.shape[0])  # Frame indices
            samples = smpl_trans[:, axis_idx]  # Values for this axis
            # Add keyframes
            fcurve.keyframe_points.add(count=len(frames))
            fcurve.keyframe_points.foreach_set(
                "co", [x for co in zip(frames, samples) for x in co]
            )
            fcurve.update()

    output_path = f"{output_folder}/{Path(pkl_name).stem}.fbx"
    bpy.ops.export_scene.fbx(filepath=output_path)
    print(f"Exported FBX to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument("--input_pkl_base", type=str, required=True)
    parser.add_argument("--fbx_source_path", type=str, required=True)
    parser.add_argument("--output_base", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fix_offset", action="store_true")
    args = parser.parse_args()

    output_folder = args.output_base if args.output_base else args.input_pkl_base
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    smpl_objects = SmplObjects(args.input_pkl_base)

    for pkl_name, smpl_params in smpl_objects:
        try:
            add_animation(
                pkl_name,
                args.fbx_source_path,
                smpl_params,
                output_folder,
                args.fps,
                args.fix_offset,
            )
        except Exception as e:
            print(f"Error processing {pkl_name}: {e}")
