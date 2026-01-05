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


import torch
from mhr.mhr import MHR
import trimesh
import open3d as o3d

torch.manual_seed(0)

def visualize_open3d(verts, faces):
    """
    verts: (V, 3) torch.Tensor or np.ndarray
    faces: (F, 3) np.ndarray
    """
    if torch.is_tensor(verts):
        verts = verts.detach().cpu().numpy()

    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype("int32"))

    mesh.compute_vertex_normals()

    # Optional but helpful
    mesh.paint_uniform_color([0.8, 0.7, 0.6])
    verts_np = verts.detach().cpu().numpy() if torch.is_tensor(verts) else verts
    center = verts_np.mean(axis=0)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100.0,
        origin=[0, 0, 0]   # WORLD ORIGIN
    )


    o3d.visualization.draw_geometries(
        [mesh, frame],
        window_name="MHR Mesh",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )

def mesh_o3d(v, faces, color):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(v.detach().cpu().numpy())
    m.triangles = o3d.utility.Vector3iVector(faces.astype("int32"))
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    return m

def _prepare_input_data(batch_size: int) -> torch.Tensor:
    identity_coeffs = 0.8 * torch.randn(batch_size, 45).cpu()
    model_parameters = 0.5 * (torch.rand(batch_size, 204) - 0.5).cpu()
    face_expr_coeffs = 0.3 * torch.randn(batch_size, 72).cpu()
    return identity_coeffs, model_parameters, face_expr_coeffs

def run():
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)
    faces = mhr_model.character.mesh.faces

    batch_size = 256
    identity_coeffs, model_parameters, face_expr_coeffs = _prepare_input_data(batch_size)
    
    with torch.no_grad():
        verts, skel_state, joint_parameters = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)

    # ---- Open3D visualization ----
    visualize_open3d(verts[132], faces)

    # sanity check
    model_parameters[:, :3] = torch.tensor([0., 0., 0.])
    v0, _, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
    model_parameters[:, :3] = torch.tensor([0., 0., 1000000])
    v1, _, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
    print((v0 - v1).abs().max())

    # forward 1
    mp0 = model_parameters.clone()
    mp0[:, :3] = torch.tensor([0., 0., 0.])
    with torch.no_grad():
        v0, _, _ = mhr_model(identity_coeffs, mp0, face_expr_coeffs)

    # forward 2
    mp1 = model_parameters.clone()
    mp1[:, :3] = torch.tensor([0., 0., 1.57])
    with torch.no_grad():
        v1, _, _ = mhr_model(identity_coeffs, mp1, face_expr_coeffs)

    m0 = mesh_o3d(v0[0], faces, [1, 0, 0])   # red
    m1 = mesh_o3d(v1[0], faces, [0, 1, 0])   # green

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0,0,0])
    o3d.visualization.draw_geometries([m0, m1, frame], mesh_show_back_face=True)


    # mesh = trimesh.Trimesh(vertices=verts[0].numpy(), faces=mhr_model.character.mesh.faces, process=False)
    # output_mesh_path = "./test.ply"
    # mesh.export(output_mesh_path)
    # print(f"Saved example MHR mesh to {output_mesh_path}")

def compare_with_torchscript_model():
    print("Comparing MHR model with TorchScripted model.")
    scripted_model = torch.jit.load("./assets/mhr_model.pt")
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

    batch_size = 128
    identity_coeffs, model_parameters, face_expr_coeffs = _prepare_input_data(batch_size)

    with torch.no_grad():
        verts, _, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
        verts_ts, _ = scripted_model(identity_coeffs, model_parameters, face_expr_coeffs)
        print(f"Averge per-vertex offsets {torch.abs(verts - verts_ts).mean()} cm.")
        print(f"Max per-vertex offsets {torch.abs(verts - verts_ts).max()} cm.")

if __name__ == "__main__":
    run()
    compare_with_torchscript_model()
