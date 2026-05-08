import sys
import torch
from mhr.mhr import MHR
import open3d as o3d

torch.manual_seed(0)

def run(updates):
    # Initialize MHR model with lowest LOD
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=0)
    faces = mhr_model.character.mesh.faces

    # Prepare default data: batch_size=1, all zeros for default pose
    identity_coeffs = torch.zeros(1, 45)
    model_parameters = torch.zeros(1, 204)  # all pose params zero
    face_expr_coeffs = torch.zeros(1, 72)

    # Apply updates from command line
    if len(updates) % 2 != 0:
        print("Invalid input: must be even number of arguments (pairs of index value).")
        return
    for i in range(0, len(updates), 2):
        try:
            idx = int(updates[i])
            val = float(updates[i+1])
            if 0 <= idx <= 129:
                model_parameters[0, 6 + idx] = val
            else:
                print(f"Index {idx} out of range 0-129.")
                return
        except ValueError:
            print("Invalid number in arguments.")
            return

    # Compute mesh
    with torch.no_grad():
        verts, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
    verts = verts / 100  # scale to meters

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts[0].detach().cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype("int32"))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # gray

    # Show in visualizer
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    o3d.visualization.draw_geometries([mesh, frame], mesh_show_back_face=True)

if __name__ == "__main__":
    run(sys.argv[1:])
