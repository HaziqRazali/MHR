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

import sys
import torch
import numpy as np
from mhr.mhr import MHR
import trimesh
import open3d as o3d

torch.manual_seed(0)

def mesh_o3d(v, faces, color):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(v.detach().cpu().numpy())
    m.triangles = o3d.utility.Vector3iVector(faces.astype("int32"))
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    return m

def _prepare_input_data(batch_size: int) -> torch.Tensor:
    identity_coeffs  = 0.8 * torch.randn(batch_size, 45).cpu()
    model_parameters = 0.5 * (torch.rand(batch_size, 204) - 0.5).cpu()
    face_expr_coeffs = 0.3 * torch.randn(batch_size, 72).cpu()
    return identity_coeffs, model_parameters, face_expr_coeffs

def run():

    # initialize MHR model
    mhr_model       = MHR.from_files(device=torch.device("cpu"), lod=1)
    mhr_model_v2    = torch.jit.load("mhr_model_v2.pt", map_location='cpu')
    faces           = mhr_model.character.mesh.faces

    # # # # # # # # #
    # sanity checks #
    # # # # # # # # #
    
    if 0:
        # This is the output of the forward pass
        # List all 127 joint names ===
        # Anchors:       0 body_world, 1 root
        # L leg:         2-8 (upleg, lowleg, foot, talocrural, subtalar, transversetarsal, ball) + 9-17 twist procs
        # R leg:        18-24 (upleg, lowleg, foot, talocrural, subtalar, transversetarsal, ball) + 25-33 twist procs
        # Spine:        34-37 (c_spine0, c_spine1, c_spine2, c_spine3)
        # R arm:        38-42 (clavicle, uparm, lowarm, wrist_twist, wrist) + 65-73 twist procs
        # R hand:       43-64 (pinky0-3, ring1-3, middle1-3, index1-3, thumb0-3 + nulls)
        # L arm:        74-78 (clavicle, uparm, lowarm, wrist_twist, wrist) + 101-109 twist procs
        # L hand:       79-100 (pinky0-3, ring1-3, middle1-3, index1-3, thumb0-3 + nulls)
        # Head/face:   110-126 (neck, neck_twist, head, jaw, teeth, tongue0-4, eyes, nulls)
        print(len(mhr_model.character_torch.skeleton.joint_names))
        for i, name in enumerate(mhr_model.character_torch.skeleton.joint_names):
            print(i, name)
        sys.exit()

    if 0:
        # This is the input to the forward pass
        # Total params = 321 total parameters across the model (after with_blend_shape())
        # 0-5:      rigid parameters (global translation + rotation)     [pt.rigid_parameters]
        # 6-135:    pose parameters (130 body joint angles)               NOTE: pt.pose_parameters spans 0-135 (includes rigid), so use pt.pose_parameters with [:6]=False to isolate body joints
        # 136-203:  scale parameters (68 mesh deformation values)         [pt.scaling_parameters]
        # 204-248:  identity blendshapes (45 coefficients)                [pt.blend_shape_parameters, first 45]
        # 249-320:  facial expression blendshapes (72 coefficients)       [pt.face_expression_parameters]
        # Ordering of 204-320 is set by set_blendshape_parameter_sets() in mhr/mhr.py
        # In mhr_head.py, body_pose_params refers specifically to indices 6:136 (the 130 joint angles)
        # but the full model_parameters [0:204] is passed to MHR forward
        # See: file:///home/haziq/MHR/mhr/mhr.py (set_blendshape_parameter_sets) for blendshape ordering
        #      file:///home/haziq/MHR/mhr/mhr.py (MHR.forward) for how model_parameters is padded and consumed
        #      file:///home/haziq/sam-3d-body/sam_3d_body/models/heads/mhr_head.py (mhr_forward) for usage
        
        # param_transform = mhr_model.character.parameter_transform
        # print("Total params:", param_transform.size)    
        # Total params: 321

        # print("\nDir of param_transform:")
        # print([attr for attr in dir(param_transform) if not attr.startswith('_')])
        # ['add_parameter_set', 'all_parameters', 'apply', 'blend_shape_parameters',
        #  'face_expression_parameters', 'find_parameters', 'inverse', 'names',
        #  'no_parameters', 'parameter_sets', 'parameters_for_joints', 'pose_parameters',
        #  'rigid_parameters', 'scaling_parameters', 'size', 'transform']
        pass

    if 0:
        # This verifies the model_parameters → joint_parameters mapping via FK, confirming the expected parameter breakdown and joint outputs
        #
        # Input:    321-dim parameter vector (the full character state)
        # 0-5:      rigid parameters (global translation + rotation)
        # 6-135:    pose parameters (130 body joint angles)
        # 136-203:  scale parameters (68 mesh deformation values)
        # 204-248:  identity blendshapes (45 coefficients)
        # 249-320:  facial expression blendshapes (72 coefficients)
        #
        # Output: 889-dim joint parameters (127 joints × 7 params per joint)
        # The 7 params per joint = [tx, ty, tz, rot_x, rot_y, rot_z, scale_log2]
        # See: mhr/mhr.py
        dummy_input = torch.zeros(1, 321)
        try:
            joint_params = mhr_model.character_torch.model_parameters_to_joint_parameters(dummy_input)
            print("\nJoint params shape:", joint_params.shape)  # Should be [1, 889] = 127*7
        except Exception as e:
            print("Error:", e)
        pass

    if 0:
        # Map pose params → joints (130-d body_pose_params only) ===
        # Shows which of the 130 pose params (indices 6-135) drive each articulated joint
        # Outputs both global param indices and local indices within body_pose_params (0-129)
        # Quick lookup: body_pose_params[i] mainly affects these joints (shared params repeat):
        # 0: c_spine0
        # 1: c_spine0,c_spine1,c_spine2
        # 2: c_spine0
        # 3: c_spine0,c_spine1,c_spine2
        # 4: c_spine0
        # 5: c_spine0,c_spine1
        # 6: c_spine1
        # 7: c_spine1,c_spine2,c_spine3
        # 8: c_spine1
        # 9: c_spine1,c_spine2,c_spine3
        # 10: c_spine1
        # 11: c_spine1,c_spine2,c_spine3
        # 12: c_spine2
        # 13: c_spine2
        # 14: c_spine2
        # 15: c_spine3
        # 16: c_spine3
        # 17: c_spine3
        # 18: c_neck
        # 19: c_neck
        # 20: c_neck
        # 21: c_head
        # 22: c_head
        # 23: c_head
        # 24: r_clavicle
        # 25: r_clavicle
        # 26: r_clavicle
        # 27: r_uparm
        # 28: r_uparm
        # 29: r_uparm
        # 30: r_lowarm
        # 31: r_wrist_twist
        # 32: r_wrist
        # 33: r_wrist
        # 34: l_clavicle
        # 35: l_clavicle
        # 36: l_clavicle
        # 37: l_uparm
        # 38: l_uparm
        # 39: l_uparm
        # 40: l_lowarm
        # 41: l_wrist_twist
        # 42: l_wrist
        # 43: l_wrist
        # 44: r_upleg
        # 45: r_upleg
        # 46: r_upleg
        # 47: r_lowleg
        # 48: r_foot
        # 49: r_talocrural
        # 50: r_subtalar
        # 51: r_transversetarsal
        # 52: r_ball
        # 53: l_upleg
        # 54: l_upleg
        # 55: l_upleg
        # 56: l_lowleg
        # 57: l_foot
        # 58: l_talocrural
        # 59: l_subtalar
        # 60: l_transversetarsal
        # 61: l_ball
        # 62: r_thumb0
        # 63: r_thumb0
        # 64: r_thumb1
        # 65: r_thumb1
        # 66: r_thumb1
        # 67: r_thumb2
        # 68: r_thumb3
        # 69: r_index1
        # 70: r_ring1
        # 71: r_pinky1
        # 72: r_middle1
        # 73: r_index1
        # 74: r_index2
        # 75: r_index3
        # 76: r_middle1
        # 77: r_middle2
        # 78: r_middle3
        # 79: r_ring1
        # 80: r_ring2
        # 81: r_ring3
        # 82: r_pinky1
        # 83: r_pinky2
        # 84: r_pinky3
        # 85: r_index1
        # 86: r_ring1
        # 87: r_pinky1
        # 88: r_middle1
        # 89: l_thumb0
        # 90: l_thumb0
        # 91: l_thumb1
        # 92: l_thumb1
        # 93: l_thumb1
        # 94: l_thumb2
        # 95: l_thumb3
        # 96: l_index1
        # 97: l_ring1
        # 98: l_pinky1
        # 99: l_middle1
        # 100: l_index1
        # 101: l_index2
        # 102: l_index3
        # 103: l_middle1
        # 104: l_middle2
        # 105: l_middle3
        # 106: l_ring1
        # 107: l_ring2
        # 108: l_ring3
        # 109: l_pinky1
        # 110: l_pinky2
        # 111: l_pinky3
        # 112: l_index1
        # 113: l_ring1
        # 114: l_pinky1
        # 115: l_middle1
        # 116: l_foot
        # 117: l_subtalar
        # 118: l_talocrural
        # 119: l_ball
        # 120: r_foot
        # 121: r_subtalar
        # 122: r_talocrural
        # 123: r_ball
        # 124: c_spine1,c_spine2,c_spine3,c_neck
        # 125: c_head
        # 126: r_uparm,l_uparm (shared shoulder twist)
        # 127: r_lowarm,r_wrist_twist,l_lowarm,l_wrist_twist (shared forearm twist)
        # 128: r_upleg,l_upleg (shared hip twist)
        # 129: r_lowleg,r_foot,l_lowleg,l_foot (shared knee/ankle)
        pose_mask = param_transform.pose_parameters.clone()
        pose_mask[:6] = False          # drop rigid
        pose_mask[136:] = False        # drop scale + blendshapes
        print("\nPose indices per articulated joint (body_pose_params 0-129):")
        for j_idx, j_name in enumerate(mhr_model.character_torch.skeleton.joint_names):
            if '_proc' in j_name or '_null' in j_name:
                continue
            joint_mask = param_transform.parameters_for_joints([j_idx])
            pose_indices = torch.where(joint_mask & pose_mask)[0]
            if pose_indices.numel() == 0:
                continue
            local_pose = (pose_indices - 6).tolist()  # shift to 0-129
            print(f"{j_idx:3d} {j_name:25s}: global {pose_indices.tolist()} -> body_pose_params {local_pose}")

    if 0:
        # === Full sweep: test all 130 body_pose_params slots ===
        # For each slot 0-129: set to 0.3, zero rest, print driven joints, assert non-driven joints stay ~0
        print("\n=== Testing all 130 body_pose_params slots ===")
        mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)
        pt = mhr_model.character.parameter_transform
        joint_names = mhr_model.character_torch.skeleton.joint_names

        for slot in range(130):
            global_idx = slot + 6
            
            # Zero everything, set just this slot to 0.3
            model_params = torch.zeros(1, 321)
            model_params[0, global_idx] = 0.3
            
            joint_params = mhr_model.character_torch.model_parameters_to_joint_parameters(model_params)
            joint_params = joint_params.view(127, 7)
            
            print(f"\nbody_pose_params[{slot}] (global {global_idx}):")
            
            affected_count = 0
            unaffected_rot_sum = 0.0
            for j in range(len(joint_names)):
                jmask = pt.parameters_for_joints([j])
                rx, ry, rz = joint_params[j, 3].item(), joint_params[j, 4].item(), joint_params[j, 5].item()
                rot_mag = (rx**2 + ry**2 + rz**2) ** 0.5
                
                if jmask[global_idx]:
                    # This joint should be driven by this slot
                    affected_count += 1
                    print(f"  {j:3d} {joint_names[j]:25s} rot=({rx:7.4f}, {ry:7.4f}, {rz:7.4f})")
                else:
                    # This joint should NOT be driven—accumulate for sanity check
                    unaffected_rot_sum += abs(rx) + abs(ry) + abs(rz)
            
            if affected_count == 0:
                print("  (no joints driven by this slot)")
            
            print(f"  Unaffected joints rot sum: {unaffected_rot_sum:.2e} (should be ~0)")
    
    # # # # # # # # # # # # #
    # end of sanity checks  #
    # # # # # # # # # # # # #

    #################### prepare data
    identity_coeffs, model_parameters, face_expr_coeffs = _prepare_input_data(batch_size=256)

    # model_parameters breakdown (0-203, 204 total):
    # 0-5:      rigid parameters (global translation x,y,z + rotation x,y,z in axis-angle)
    # 6-135:    pose parameters (130 body joint angles in axis-angle, local to parent)
    # 136-203:  scale parameters (68 mesh deformation/scaling values)
    print(f"model_parameters.shape: {model_parameters.shape}")  # [256, 204]
    
    #################### forward pass
    print(f"Sanity check: zero out left arm/hand and left leg chains")
    # See: file:///home/haziq/sam-3d-body/sam_3d_body/MHR/mhr.py
    # See: file:///home/haziq/sam-3d-body/sam_3d_body/models/heads/mhr_head.py
    
    # Test: zero out left arm/hand and left leg chains (no rotation from shoulder/hip onwards)
    # Left arm in body_pose_params: 34-43 (l_clavicle through l_wrist)
    # Left hand in body_pose_params: 89-115 (l_thumb0 through l_middle1)
    # Left leg in body_pose_params: 53-61 (l_upleg through l_ball), 116-119 (extra foot)
    # Shared params: 126-127 (shoulder/forearm twist), 128-129 (hip/knee twist)
    # In model_parameters: add 6 to body_pose_params indices
    mp_left_zero = model_parameters.clone()
    mp_left_zero[:, 40:50]      = 0 # left arm (body_pose_params 34-43 + 6)
    mp_left_zero[:, 95:122]     = 0 # left hand (body_pose_params 89-115 + 6)
    mp_left_zero[:, 59:68]      = 0 # left leg (body_pose_params 53-61 + 6)
    mp_left_zero[:, 122:126]    = 0 # left foot extra (body_pose_params 116-119 + 6)
    mp_left_zero[:, 132:136]    = 0 # shared shoulder/forearm/hip/knee twist (126-129 + 6)
    
    with torch.no_grad():
        v_zero, skel_state_zero     = mhr_model(identity_coeffs, mp_left_zero, face_expr_coeffs)
        v_normal, skel_state_normal = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
        v_zero      = v_zero / 100
        v_normal    = v_normal / 100

        v_zero_v2, skel_state_zero_v2       = mhr_model_v2(identity_coeffs, mp_left_zero, face_expr_coeffs)
        v_normal_v2, skel_state_normal_v2   = mhr_model_v2(identity_coeffs, model_parameters, face_expr_coeffs)
        v_zero_v2       = v_zero_v2 / 100
        v_normal_v2     = v_normal_v2 / 100

    idx         = 132
    m_zero      = mesh_o3d(v_zero[idx], faces, [1, 0, 0])       # red - left arm zeroed
    m_zero_v2   = mesh_o3d(v_zero_v2[idx], faces, [0, 0, 1])    # blue - v2 zeroed
    m_normal    = mesh_o3d(v_normal[idx], faces, [0, 1, 0])     # green - normal pose
    m_normal_v2 = mesh_o3d(v_normal_v2[idx], faces, [1, 1, 0])  # yellow - v2 normal
    frame       = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    geometries  = [m_zero, m_zero_v2, m_normal, m_normal_v2]
    for i in range(1, len(geometries) + 1):
        o3d.visualization.draw_geometries(geometries[:i] + [frame], mesh_show_back_face=True)
    

    #################### check skeleton state to joint parameters mapping
    print(f"Checking skeleton_state_to_joint_parameters mapping")
    
    joint_parameters_zero   = mhr_model_v2.skeleton_state_to_joint_parameters(skel_state_zero)
    skel_state_zero_back    = mhr_model.character_torch.joint_parameters_to_skeleton_state(joint_parameters_zero)
    print(f"Max diff skel_state_zero -> joint_params -> skel_state: "f"{(skel_state_zero_back - skel_state_zero).abs().max()}")
    joint_parameters_normal = mhr_model_v2.skeleton_state_to_joint_parameters(skel_state_normal)
    skel_state_normal_back  = mhr_model.character_torch.joint_parameters_to_skeleton_state(joint_parameters_normal)
    print(f"Max diff skel_state_normal -> joint_params -> skel_state: "f"{(skel_state_normal_back - skel_state_normal).abs().max()}")
    print() 

    # inspect joint parameters
    euler_angles_zero = joint_parameters_zero.view(-1, 127, 7)[:, :, 3:6]  # [B, 127, 3]
    print(euler_angles_zero[0, 75, :])  # print first sample
    euler_angles_normal = joint_parameters_normal.view(-1, 127, 7)[:, :, 3:6]  # [B, 127, 3]
    print(euler_angles_normal[0, 75, :])  # print first sample
    sys.exit()

    #################### numerical sanity check
    print(f"Sanity check: model shifted by translation")
    model_parameters[:, :3]     = torch.tensor([0., 0., 0.])        # set translation to 0
    model_parameters[:, 3:6]    = torch.tensor([0., 0., 0.])        # set rotation to 0
    vertex0, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)

    model_parameters[:, :3]     = torch.tensor([0., 0., 100])       # set translation to a large value
    model_parameters[:, 3:6]    = torch.tensor([0., 0., 0.])        # set rotation to 0
    vertex1, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
    print(f"{(vertex0 - vertex1).abs().max()} (should be 100)")
    print()

    print(f"Sanity check with mhr_model_v2: model shifted by translation of 100")
    model_parameters[:, :3]     = torch.tensor([0., 0., 0.])        # set translation to 0
    model_parameters[:, 3:6]    = torch.tensor([0., 0., 0.])        # set rotation to 0
    vertex0_v2, _ = mhr_model_v2(identity_coeffs, model_parameters, face_expr_coeffs)

    model_parameters[:, :3]     = torch.tensor([0., 0., 100])       # set translation to a large value
    model_parameters[:, 3:6]    = torch.tensor([0., 0., 0.])        # set rotation to 0
    vertex1_v2, _ = mhr_model_v2(identity_coeffs, model_parameters, face_expr_coeffs)
    print(f"{(vertex0_v2 - vertex1_v2).abs().max()} (should be 100)")
    print()

    print(f"{(vertex0 - vertex0_v2).abs().max()} (should be 0)")
    print(f"{(vertex1 - vertex1_v2).abs().max()} (should be 0)")
    print()

    #################### visual sanity check
    print(f"Sanity check: model rotated by 90 degrees about Z axis")

    # forward 1
    mp0 = model_parameters.clone()
    mp0[:, :3]  = torch.tensor([0., 0., 10.])  # set translation to 1000
    mp0[:, 3:6] = torch.tensor([0., 0., 0.])     # set rotation to 0
    with torch.no_grad():
        v0, _ = mhr_model(identity_coeffs, mp0, face_expr_coeffs)
        v0 = v0 / 100

    # forward 2
    mp1 = model_parameters.clone()
    mp1[:, :3]  = torch.tensor([0., 0., 0.])     # set translation to 0
    mp1[:, 3:6] = torch.tensor([0., 0., 1.57])   # set rotation to 0
    with torch.no_grad():
        v1, _ = mhr_model(identity_coeffs, mp1, face_expr_coeffs)
        v1 = v1 / 100

    m0 = mesh_o3d(v0[idx], faces, [1, 0, 0])   # red
    m1 = mesh_o3d(v1[idx], faces, [0, 1, 0])   # green

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    o3d.visualization.draw_geometries([m0, m1, frame], mesh_show_back_face=True)
    np.save("faces.npy", faces)

    #################### mapping sanity check

def compare_with_torchscript_model():
    print("Comparing MHR model with TorchScripted model.")
    #scripted_model = torch.jit.load("./assets/mhr_model.pt")
    scripted_model = torch.jit.load("mhr_model_v2.pt")
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

    batch_size = 128
    identity_coeffs, model_parameters, face_expr_coeffs = _prepare_input_data(batch_size)

    with torch.no_grad():
        verts, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
        verts_ts, _ = scripted_model(identity_coeffs, model_parameters, face_expr_coeffs)
        print(f"Averge per-vertex offsets {torch.abs(verts - verts_ts).mean()} cm.")
        print(f"Max per-vertex offsets {torch.abs(verts - verts_ts).max()} cm.")

if __name__ == "__main__":
    run()
    compare_with_torchscript_model()
