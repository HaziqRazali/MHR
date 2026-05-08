import torch
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import os
from mhr.mhr import MHR
from conversion import Conversion
import smplx
import trimesh

sys.path.append('/media/haziq/Haziq/mocap/my_scripts/imar_vision_datasets_tools/')
from util.dataset_util import aa_to_rotmat

# Set headless backend for pyrender
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

try:
    import pyrender
    import imageio
    PYRENDER_AVAILABLE = True
except ImportError as e:
    print(f"pyrender not available: {e}")
    PYRENDER_AVAILABLE = False

"""
python smpl_to_mhr.py \
--smplx_filename /media/haziq/Haziq/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
--mhr_filename smpl_to_mhr.npz \
--mhr_video_filename smpl_to_mhr.mp4
"""

# Parse arguments
parser = argparse.ArgumentParser(description='Convert SMPLX to MHR and visualize')
parser.add_argument('--smplx_filename', type=str, required=True, help='Path to SMPLX JSON file')
parser.add_argument('--mhr_filename', type=str, default='mhr_converted.npz', help='Output filename for MHR parameters (.npz)')
parser.add_argument('--mhr_video_filename', type=str, default='smplx_mhr_comparison.mp4', help='Output filename for comparison video (.mp4)')
args = parser.parse_args()

# print("Loading sample MHR data for structure reference...")
# mhr_sample = np.load("/media/haziq/Haziq/mocap/data/fit3d/train/s03/sam3d/60457274/band_pull_apart_mhr_outputs.npz", allow_pickle=True)
# for k,v in mhr_sample.items():
#     print(k, v.shape)
# print()

def rotation_matrix_to_axis_angle(rotmat):
    # rotmat: [..., 3, 3]
    shape = rotmat.shape
    rotmat = rotmat.reshape(-1, 3, 3)
    aa = []
    for r in rotmat:
        r_np = r.cpu().numpy()
        rot = Rotation.from_matrix(r_np)
        aa_vec = rot.as_rotvec()
        aa.append(aa_vec)
    aa = np.array(aa).reshape(shape[:-2] + (3,))
    return torch.tensor(aa, device=rotmat.device, dtype=rotmat.dtype)

# Initialize models
mhr_model = MHR.from_files(lod=1, device=torch.device("cuda:0"))
smplx_model = smplx.SMPLX(model_path="/media/haziq/Haziq/mocap/data/models_smplx_v1_1/models/smplx", gender="neutral", use_pca=False)

# Create converter
converter = Conversion(
    mhr_model=mhr_model,
    smpl_model=smplx_model,
    method="pytorch"  # or "pymomentum"
)

smplx_filename = args.smplx_filename
if smplx_filename.endswith('.json'):
    smplx_outputs = json.load(open(smplx_filename, "r"))
elif smplx_filename.endswith('.npz'):
    smplx_outputs = dict(np.load(smplx_filename, allow_pickle=True))
else:
    raise ValueError(f"Unsupported file format: {smplx_filename}. Expected .json or .npz")

if "humaneva" in args.smplx_filename:
                        
    #for k, v in world_smplx_params.items():
    #    if isinstance(v, (np.ndarray, torch.Tensor)):
    #        print(k, tuple(v.shape))
    #sys.exit()

    smplx_outputs["transl"]        = smplx_outputs.pop("trans")       # [t, 3]
    smplx_outputs["global_orient"] = smplx_outputs.pop("root_orient") # [t, 3]
    smplx_outputs["body_pose"]     = smplx_outputs.pop("pose_body")   # [t, 63]
    smplx_outputs["betas"]         = smplx_outputs["betas"][:10]      # [3]
    smplx_outputs["pose_hand"]     = smplx_outputs.pop("pose_hand")
    smplx_outputs["pose_jaw"]      = smplx_outputs.pop("pose_jaw")
    smplx_outputs["pose_eye"]      = smplx_outputs.pop("pose_eye")

    # convert to format compatible with the SMPLXHelper
    num_mocap_frames = smplx_outputs["transl"].shape[0]
    smplx_outputs["global_orient"] = smplx_outputs["global_orient"][:,None,:]                             # [t, 1, 3]
    smplx_outputs["global_orient"] = aa_to_rotmat(smplx_outputs["global_orient"])                         # [t, 1, 3, 3]
    smplx_outputs["body_pose"]     = np.reshape(smplx_outputs["body_pose"],[num_mocap_frames, 21, 3])     # [t, 21 ,3]
    smplx_outputs["body_pose"]     = aa_to_rotmat(smplx_outputs["body_pose"])                             # [t, 21, 3, 3]
    smplx_outputs["betas"]         = smplx_outputs["betas"][None,:10].repeat(num_mocap_frames,0)          # [t, 10]

    smplx_params = {
        'global_orient': torch.tensor(smplx_outputs['global_orient'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
        'body_pose': torch.tensor(smplx_outputs['body_pose'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
        'betas': torch.tensor(smplx_outputs['betas'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
    }
    for key in ['global_orient', 'body_pose']:
        aa = rotation_matrix_to_axis_angle(smplx_params[key])
        smplx_params[key] = aa.reshape(aa.shape[0], -1)
    smplx_params['left_hand_pose']  = torch.zeros((num_mocap_frames, 15*3), device=torch.device("cuda:0"), dtype=torch.float32)
    smplx_params['right_hand_pose'] = torch.zeros((num_mocap_frames, 15*3), device=torch.device("cuda:0"), dtype=torch.float32)
    smplx_params['jaw_pose']       = torch.zeros((num_mocap_frames, 3), device=torch.device("cuda:0"), dtype=torch.float32)
    smplx_params['leye_pose']      = torch.zeros((num_mocap_frames, 3), device=torch.device("cuda:0"), dtype=torch.float32)
    smplx_params['reye_pose']      = torch.zeros((num_mocap_frames, 3), device=torch.device("cuda:0"), dtype=torch.float32)
    for k,v in smplx_params.items():
        print(k, v.shape)

else:
    smplx_params = {
        'global_orient': torch.tensor(smplx_outputs['global_orient'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
        'body_pose': torch.tensor(smplx_outputs['body_pose'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
        'left_hand_pose': torch.tensor(smplx_outputs['left_hand_pose'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
        'right_hand_pose': torch.tensor(smplx_outputs['right_hand_pose'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
        'jaw_pose': torch.tensor(smplx_outputs['jaw_pose'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
        'leye_pose': torch.tensor(smplx_outputs['leye_pose'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
        'reye_pose': torch.tensor(smplx_outputs['reye_pose'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
        'betas': torch.tensor(smplx_outputs['betas'], device=torch.device("cuda:0"), dtype=torch.float32).squeeze(0),
    }
    for key in ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose']:
        aa = rotation_matrix_to_axis_angle(smplx_params[key])
        smplx_params[key] = aa.reshape(aa.shape[0], -1)
    for k,v in smplx_params.items():
        print(k, v.shape)

#for k,v in smplx_params.items():
#    smplx_params[k] = v[:10]    # Take only first 10 frames for testing

# Convert SMPLX to MHR
results = converter.convert_smpl2mhr(
    smpl_parameters=smplx_params,
    single_identity=True,
    return_mhr_meshes=True,
    return_mhr_parameters=True,
    is_tracking=True,
)

def print_structure(obj, path="", max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            shape_info = f" shape={v.shape}" if hasattr(v, 'shape') else ""
            print(f"{path}.{k}: {type(v).__name__}{shape_info}")
            if isinstance(v, (dict, list, tuple)) and current_depth < max_depth:
                print_structure(v, f"{path}.{k}", max_depth, current_depth + 1)
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        shape_info = f" shape={obj[0].shape}" if hasattr(obj[0], 'shape') else ""
        print(f"{path}[0]: {type(obj[0]).__name__}{shape_info}")
        if isinstance(obj[0], (dict, list, tuple)) and current_depth < max_depth:
            print_structure(obj[0], f"{path}[0]", max_depth, current_depth + 1)
    else:
        shape_info = f" shape={obj.shape}" if hasattr(obj, 'shape') else ""
        print(f"{path}: {type(obj).__name__}{shape_info}")

#print("\nRecursive structure:")
#print_structure(vars(results))
# see file:///home/haziq/sam-3d-body/sam_3d_body/models/heads/mhr_head.py
# 204 = [global trans (3), global rot (3), body_pose_params (130), scales (68)]
# Recursive structure:
# .result_meshes: list
# .result_meshes[0]: Trimesh
# .result_vertices: NoneType
# .result_parameters: dict
# .result_parameters.lbs_model_params: Tensor shape=torch.Size([10, 204])
# .result_parameters.identity_coeffs: Tensor shape=torch.Size([10, 45])
# .result_parameters.face_expr_coeffs: Tensor shape=torch.Size([10, 72])
# .result_errors: ndarray shape=(10,)

# Extract MHR parameters
mhr_params = results.result_parameters

# Rename and reshape to match format required by mocap_mainloader
mhr_formatted = {
    'expr_params': mhr_params['face_expr_coeffs'].detach().cpu().numpy(),
    'shape_params': mhr_params['identity_coeffs'].detach().cpu().numpy(),
    'global_trans': mhr_params['lbs_model_params'][:, :3].detach().cpu().numpy(),
    'global_orient': mhr_params['lbs_model_params'][:, 3:6].detach().cpu().numpy(),
    'body_pose_params': mhr_params['lbs_model_params'][:, 6:136].detach().cpu().numpy(),
}
#for k,v in mhr_formatted.items():
#    print(k, v.shape)

np.savez(args.mhr_filename, **mhr_formatted)

#print(f"MHR parameters saved to {args.mhr_filename}")

mhr_model    = torch.jit.load("/home/haziq/MHR/mhr_model_v2.pt", map_location='cpu')
shape_params = torch.zeros(mhr_formatted['shape_params'].shape[0], 45)
model_parameters = torch.cat([
    torch.tensor(mhr_formatted["global_trans"]),
    torch.tensor(mhr_formatted["global_orient"]),
    torch.tensor(mhr_formatted["body_pose_params"]),
    torch.zeros(mhr_formatted['shape_params'].shape[0], 68)
], dim=1)
face_expr_coeffs = torch.zeros(mhr_formatted['shape_params'].shape[0], 72)
mhr_vertices, mhr_faces = mhr_model(shape_params, model_parameters, face_expr_coeffs)
mhr_vertices *= 0.01
mhr_mesh_zero_list = []
for i in range(mhr_vertices.shape[0]):
    mhr_mesh_zero = trimesh.Trimesh(vertices=mhr_vertices[i].detach().cpu().numpy(), faces=results.result_meshes[0].faces)
    mhr_mesh_zero_list.append(mhr_mesh_zero)

# Get SMPLX meshes for visualization
smplx_meshes, _ = converter._smpl_para2mesh(smplx_params, return_mesh=True)
mhr_meshes = results.result_meshes

# Scale MHR meshes to match SMPLX scale (MHR is in cm, SMPLX in m)
for mesh in mhr_meshes:
    mesh.vertices *= 0.01  # Convert cm to m

# Create scene
scene = pyrender.Scene()

# Add camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.eye(4)
camera_pose[2, 3] = 3  # Move camera back
scene.add(camera, pose=camera_pose)

# Add light
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
scene.add(light, pose=camera_pose)

# Offscreen renderer
r = pyrender.OffscreenRenderer(1920, 720)  # Wider for three side-by-side

images = []
num_frames = min(len(smplx_meshes), len(mhr_meshes))

for i in range(num_frames):
    # Clear scene
    scene.clear()
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    
    # Add SMPLX mesh (left side)
    smplx_pose = np.eye(4)
    smplx_pose[0, 3] = -2  # Translate left
    scene.add(pyrender.Mesh.from_trimesh(smplx_meshes[i]), pose=smplx_pose)
    
    # Add MHR mesh (middle)
    mhr_pose = np.eye(4)
    mhr_pose[0, 3] = 0  # Center
    scene.add(pyrender.Mesh.from_trimesh(mhr_meshes[i]), pose=mhr_pose)
    
    # Add MHR zero mesh (right side)
    mhr_zero_pose = np.eye(4)
    mhr_zero_pose[0, 3] = 2  # Translate right
    scene.add(pyrender.Mesh.from_trimesh(mhr_mesh_zero_list[i]), pose=mhr_zero_pose)
    
    # Render
    color, _ = r.render(scene)
    images.append(color)

# Save as video
imageio.mimsave(args.mhr_video_filename, images, fps=30)
# print(f"Video saved as {args.mhr_video_filename}")
