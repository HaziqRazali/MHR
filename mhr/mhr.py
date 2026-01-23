# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

from pathlib import Path
from typing import Literal

import numpy as np
import pymomentum.geometry as pym_geometry

import pymomentum.torch.character as torch_character

import torch

from .io import (
    get_corrective_activation_path,
    get_default_asset_folder,
    get_mhr_blendshapes_path,
    get_mhr_fbx_path,
    get_mhr_model_path,
    has_pose_corrective_blendshapes,
    load_pose_dirs_predictor,
)
from .utils import batch6DFromXYZ

LOD = Literal[0, 1, 2, 3, 4, 5, 6]
NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = 72

import torch
from typing import Optional

def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [...,4] in [x,y,z,w]
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([rx, ry, rz, rw], dim=-1)

def _quat_inverse(q: torch.Tensor) -> torch.Tensor:
    # q: [...,4] -> inverse (conjugate / normsq)
    xyz = q[..., :3]
    w = q[..., 3:4]
    conj = torch.cat([-xyz, w], dim=-1)
    normsq = (q * q).sum(dim=-1, keepdim=True)
    return conj / (normsq + 1e-12)

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # rotate vector v by quaternion q: v' = v + 2*cross(q_xyz, cross(q_xyz, v) + q_w*v)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)

def _quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    # q: [...,4] in [x,y,z,w] -> returns [...,3] Euler angles (X, Y, Z) = (roll, pitch, yaw)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    rx = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    ry = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    rz = torch.atan2(t3, t4)

    return torch.stack([rx, ry, rz], dim=-1)

def skel_state_to_joint_parameters(
    skel_state: torch.Tensor,
    parents: torch.Tensor,
    prerot: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert global skeleton state to flattened joint parameters.

    Args:
      skel_state: Tensor[..., J, 8] with layout [tx,ty,tz, qx,qy,qz,qw, scale]
      parents: LongTensor[J] with parent indices (root uses -1)
      prerot: Optional Tensor[J,4] quaternions [x,y,z,w] (or None for identity)
      offsets: Optional Tensor[J,3] joint_translation_offsets (or None for zeros)

    Returns:
      joint_parameters: Tensor[..., J*7] flattened per-joint [tx,ty,tz, rot_x,rot_y,rot_z, log2_scale]
    """
    # shapes
    *batch_shape, J, dim8 = list(skel_state.shape)
    assert dim8 == 8, "skel_state last dim must be 8"

    device = skel_state.device
    dtype = skel_state.dtype

    # Prepare tensors
    parent_idx = torch.clamp(parents, min=0).to(device=device)
    parent_skel = skel_state.index_select(-2, parent_idx)  # [..., J, 8]

    global_t = skel_state[..., :3]
    global_q = skel_state[..., 3:7]
    global_s = skel_state[..., 7:8]

    parent_t = parent_skel[..., :3]
    parent_q = parent_skel[..., 3:7]
    parent_s = parent_skel[..., 7:8]

    # inverse parent rotation
    inv_parent_q = _quat_inverse(parent_q)

    # delta translation and local translation: rotate(inv_parent_q, global_t - parent_t) / parent_s
    delta_t = global_t - parent_t
    local_t = _quat_rotate(inv_parent_q, delta_t) / (parent_s + 1e-12)

    # local rotation and scale
    local_q = _quat_mul(inv_parent_q, global_q)
    local_s = global_s / (parent_s + 1e-12)

    # For joints with parent == -1, keep global as local (do not use computed values)
    parent_mask = (parents >= 0).to(device=device)  # [J]
    # expand mask to batch + joint dims
    expand_dims = [1] * (skel_state.ndim - 2) + [J, 1]  # e.g., for [..., J, 1]
    parent_mask_b = parent_mask.view(*([1] * (skel_state.ndim - 2)), J).to(device=device)
    parent_mask_b = parent_mask_b.expand(*((1,) * (skel_state.ndim - 2)), J).to(device=device).unsqueeze(-1)

    # Use torch.where to select computed local vs original global when parent < 0
    keep_local = parent_mask_b
    local_t = torch.where(keep_local, local_t, global_t)
    local_q = torch.where(keep_local, local_q, global_q)
    local_s = torch.where(keep_local, local_s, global_s)

    # subtract translation offsets if provided
    if offsets is None:
        translation_params = local_t
    else:
        # ensure offsets shape broadcast: [J,3] -> [1,...,J,3]
        translation_params = local_t - offsets.to(device=device, dtype=dtype)[None, ...] if skel_state.ndim > 2 else local_t - offsets.to(device=device, dtype=dtype)

    # remove prerotation if provided
    if prerot is None:
        adjusted_q = local_q
    else:
        prerot_b = prerot.to(device=device, dtype=dtype)
        # prerot_b: [J,4] -> broadcast
        adjusted_q = _quat_mul(_quat_inverse(prerot_b[None, ...] if skel_state.ndim > 2 else prerot_b), local_q)

    # convert quaternion -> euler XYZ
    rotation_params = _quat_to_euler_xyz(adjusted_q)

    # scale param: log2(local_s)
    scale_params = torch.log2(local_s.clamp(min=1e-12))

    # concat per-joint params and flatten last two dims
    joint_params = torch.cat([translation_params, rotation_params, scale_params], dim=-1)  # [..., J, 7]
    return joint_params.flatten(-2, -1)

class MHRPoseCorrectivesModel(torch.nn.Module):
    """Non-linear pose correctives model."""

    def __init__(self, pose_dirs_predictor: torch.nn.Sequential) -> None:
        super().__init__()

        # Network to predict pose correctives offsets
        self.pose_dirs_predictor = pose_dirs_predictor

    def _pose_features_from_joint_params(
        self, joint_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Compute pose features, input to the pose correctives network, based on joint parameters."""

        joint_euler_angles = joint_parameters.reshape(
            joint_parameters.shape[0], -1, pym_geometry.PARAMETERS_PER_JOINT
        )[
            :, 2:, 3:6
        ]  # Extract rotations (Euler XYZ) from joint parameters, excluding the first two joints (not defining local pose)
        joint_6d_feat = batch6DFromXYZ(joint_euler_angles)
        # Setting also the elements of the matrix diagonal to 0 when there is no rotation (so everything is set to 0)
        joint_6d_feat[:, :, 0] -= 1
        joint_6d_feat[:, :, 4] -= 1
        joint_6d_feat = joint_6d_feat.flatten(1, 2)
        return joint_6d_feat

    def forward(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        """Compute pose correctives given joint parameters (local per-joint transforms)."""

        pose_6d_feats = self._pose_features_from_joint_params(joint_parameters)
        pose_corrective_offsets = self.pose_dirs_predictor(pose_6d_feats).reshape(
            pose_6d_feats.shape[0], -1, 3
        )
        return pose_corrective_offsets

class MHR(torch.nn.Module):
    """MHR body model."""

    def __init__(
        self,
        character: pym_geometry.Character,
        pose_correctives_model: MHRPoseCorrectivesModel | None,
        device: torch.device,
    ) -> None:
        super().__init__()

        # Save pose correctives model
        self.pose_correctives_model = pose_correctives_model

        # Save cpu/gpu characters
        self.character = character
        # Note that this call also instantiates the identity and face expressions model
        self.character_torch = torch_character.Character(character).to(device)

    @staticmethod
    def _create_model(
        character: pym_geometry.Character,
        blendshapes_path: str,
        corrective_activation_path: str | None,
        device: torch.device,
    ) -> "MHR":
        """Create MHR model from the given character and asset paths."""

        blendshapes_data = np.load(blendshapes_path)

        # Pose correctives model
        pose_correctives_model = None
        has_pose_correctives = (
            has_pose_corrective_blendshapes(blendshapes_data)
            and corrective_activation_path is not None
        )
        if has_pose_correctives:
            corrective_activation_data = np.load(corrective_activation_path)
            pose_correctives_model = MHRPoseCorrectivesModel(
                load_pose_dirs_predictor(
                    blendshapes_data,
                    corrective_activation_data,
                    load_with_cuda=device.type == "cuda",
                )
            )

        if pose_correctives_model is not None:
            pose_correctives_model.to(device)

        return MHR(character, pose_correctives_model, device=device)

    @staticmethod
    def from_files(
        folder: Path = get_default_asset_folder(),
        device: torch.device = "cuda",
        lod: LOD = 1,
        wants_pose_correctives: bool = True,
    ) -> "MHR":
        """Load character and model parameterization, and create full model."""

        #print(folder) # /home/haziq/MHR/assets

        # Create character by fetching rig and model parameterization paths
        fbx_path = get_mhr_fbx_path(folder, lod)
        model_path = get_mhr_model_path(folder)
        assert os.path.exists(fbx_path), f"FBX file not found at {fbx_path}"
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        # Load rig and model parameterization
        character = pym_geometry.Character.load_fbx(
            fbx_path, model_path, load_blendshapes=True
        )
        assert (
            character.blend_shape.shape_vectors.shape[0]
            == NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES
        ), f"Expected {NUM_IDENTITY_BLENDSHAPES} identity and {NUM_FACE_EXPRESSION_BLENDSHAPES} face expression blendshapes, got {character.blend_shape.shape_vectors.shape[0]}"

        n_params = character.parameter_transform.size
        character = character.with_blend_shape(
            character.blend_shape
        )  # update parameter transform to include blendshape coefficients
        # Assert number of parameters now include blendshape coefficients
        assert character.parameter_transform.size == (
            n_params + NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES
        )
        # Set parameter sets for identity / facial expressions
        set_blendshape_parameter_sets(character)

        # Retrieve correctives paths and create full model
        blendshapes_path = get_mhr_blendshapes_path(folder, lod)
        corrective_activation_path = (
            get_corrective_activation_path(folder) if wants_pose_correctives else None
        )
        assert os.path.exists(
            blendshapes_path
        ), f"Blendshapes file not found at {blendshapes_path}"
        if corrective_activation_path is not None:
            assert os.path.exists(
                corrective_activation_path
            ), f"Corrective activation file not found at {corrective_activation_path}"
        return MHR._create_model(
            character, blendshapes_path, corrective_activation_path, device
        )

    def get_num_identity_blendshapes(self) -> int:
        """Return number of identity blendshapes."""

        return NUM_IDENTITY_BLENDSHAPES

    def get_num_face_expression_blendshapes(self) -> int:
        """Return number of face expression blendshapes."""

        return NUM_FACE_EXPRESSION_BLENDSHAPES

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor | None,
        apply_correctives: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vertices given input parameters."""

        # identity_coeffs:  [b=batch_size, c=num_shape_coeff]
        # model_parameters: [b=batch_size, c=num_model_params (rigid, pose, scale)]
        # face_expr_coeffs: [b=batch_size, c=num_face_coeff]
        assert (
            len(identity_coeffs.shape) == 2
        ), f"Expected batched (n_rows >= 1) identity coeffs with {self.get_num_identity_blendshapes()} columns, got {identity_coeffs.shape}"
        if face_expr_coeffs is not None:
            # Check batch sizes of face expression coeffs and model parameters are the same
            assert (
                len(face_expr_coeffs.shape) == 2
            ), f"Expected batched (n_rows >= 1) face expressions coeffs with {self.get_num_face_expression_blendshapes()} columns, got {face_expr_coeffs.shape}"
        else:
            # Create zero padding for face expression coeffs
            face_expr_coeffs = torch.zeros(
                model_parameters.shape[0], self.get_num_face_expression_blendshapes()
            ).to(identity_coeffs)
        apply_correctives = (
            apply_correctives and self.pose_correctives_model is not None
        )

        identity_coeffs = identity_coeffs.expand(model_parameters.shape[0], -1)

        coeffs = torch.cat([identity_coeffs, face_expr_coeffs], dim=1)
        # Compute vertices in rest pose
        rest_pose = self.character_torch.blend_shape.forward(coeffs)

        # Compute joint parameters (local) and skeleton state (global)
        # We need to pass as many model parameters as the parameter transform size
        model_padding = (
            torch.zeros(
                model_parameters.shape[0],
                self.get_num_face_expression_blendshapes()
                + self.get_num_identity_blendshapes(),
            )
            .to(model_parameters)
            .requires_grad_(False)
        )

        #print(model_parameters.shape)  # [256, 204]
        #print(model_padding.shape)     # [256, 117]
        
        # Analogy: model_parameters are coefficients in a compact basis; the parameter_transform matrix reconstructs the full per-joint DOFs (like taking coefficients and constructing the full signal from basis vectors).
        joint_parameters = self.character_torch.model_parameters_to_joint_parameters(torch.concatenate((model_parameters, model_padding), axis=1))
        # goes to file:///home/haziq/anaconda3/envs/mhr/lib/python3.12/site-packages/pymomentum/torch/character.py
        # joint_parameters: [batch=256, 127 joints * 7 params]
        # Contains LOCAL transforms (relative to parent joint, not global/world space)
        # Per-joint structure: [tx, ty, tz, rot_x, rot_y, rot_z, scale_param]
        #   [:3]   : translation offset relative to parent
        #   [3:6]  : local Euler XYZ rotation angles
        #   [-1]   : log2 scale factor
        # These are fed to forward kinematics to compute global skeleton state
        
        skel_state = self.character_torch.joint_parameters_to_skeleton_state(joint_parameters)
        # skel_state: [batch=256, num_joints=127, 8 params per joint]
        # Contains GLOBAL transforms in world/character space (result of forward kinematics)
        # Per-joint structure: [tx, ty,y tz, qx, qy, qz, qw, scale]
        #   [:3]   : global joint position (not relative to parent)
        #   [3:7]  : global rotation as quaternion (not relative to parent)
        #   [7]    : global scale factor
        # Accounts for entire parent chain hierarchy via FK

        if 1:
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Debug prints to create a mapping from skel_state to joint_parameters  #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            # Inspect skeleton and extract hardcode-friendly lists for a scripted wrapper
            s = self.character_torch.skeleton
            jp = skel_state_to_joint_parameters(skel_state, s.joint_parents, prerot=s.joint_prerotations, offsets=s.joint_translation_offsets)
            diff_jp = (jp - joint_parameters).abs().max().item()
            print("max |jp - joint_parameters| =", diff_jp)
            assert diff_jp < 1e-3, f"joint params mismatch (max {diff_jp})"

            ss = self.character_torch.joint_parameters_to_skeleton_state(jp)
            diff_ss = (ss - skel_state).abs().max().item()
            print("max |ss - skel_state| =", diff_ss)
            assert diff_ss < 1e-3, f"skel_state mismatch (max {diff_ss})"

            jp = skel_state_to_joint_parameters(ss, s.joint_parents, prerot=s.joint_prerotations, offsets=s.joint_translation_offsets)        
            diff_jp = (jp - joint_parameters).abs().max().item()
            print("max |jp - joint_parameters| =", diff_jp)
            assert diff_jp < 1e-3, f"joint params mismatch (max {diff_jp})"

            ss = self.character_torch.joint_parameters_to_skeleton_state(jp)        
            diff_ss = (ss - skel_state).abs().max().item()
            print("max |ss - skel_state| =", diff_ss)
            assert diff_ss < 1e-3, f"skel_state mismatch (max {diff_ss})"
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Debug prints to create a mapping from skel_state to joint_parameters  #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # Apply pose correctives
        linear_model_unposed = rest_pose
        if apply_correctives:
            linear_model_pose_correctives = self.pose_correctives_model.forward(
                joint_parameters=joint_parameters
            )
            linear_model_unposed += linear_model_pose_correctives

        # Compute vertices
        verts = self.character_torch.skin_points(
            skel_state=skel_state, rest_vertex_positions=linear_model_unposed
        )

        return verts, skel_state


def set_blendshape_parameter_sets(character: pym_geometry.Character) -> None:
    """Utility function to discriminate between identity/facial expression blendshape parameters of a character."""

    # Check number of blendshapes is as expected
    n_shapes = character.blend_shape.n_shapes
    assert n_shapes == (NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES)

    # Set parameter set for identity
    identity_parameter_set = torch.zeros(
        character.parameter_transform.size, dtype=torch.bool
    )
    identity_parameter_set[-n_shapes : -n_shapes + NUM_IDENTITY_BLENDSHAPES] = True
    character.parameter_transform.add_parameter_set("identity", identity_parameter_set)

    # Set parameter set for facial expressions
    face_expression_parameter_set = torch.zeros(
        character.parameter_transform.size, dtype=torch.bool
    )
    face_expression_parameter_set[-NUM_FACE_EXPRESSION_BLENDSHAPES:] = True
    character.parameter_transform.add_parameter_set(
        "faceExpression", face_expression_parameter_set
    )
