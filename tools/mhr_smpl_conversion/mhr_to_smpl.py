import os
import sys
import cv2
import torch
import smplx
import numpy as np

from mhr.mhr import MHR
from conversion import Conversion

sys.path.append(os.path.expanduser('~/datasets/mocap/my_scripts/'))
from utils_draw import render_simple_pyrender

def to_torch(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=torch.float32)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------------------------------------
    # 1. Load MHR model
    # ------------------------------------------------------------
    mhr_model = MHR.from_files(lod=1, device=device)

    # ------------------------------------------------------------
    # 2. Build SMPL-X model (MATCH converter output!)
    #    - PCA hands
    #    - 6 PCA comps (converter outputs [T,6])
    # ------------------------------------------------------------
    smplx_model = smplx.SMPLX(
        model_path="/home/haziq/datasets/mocap/data/models_smplx_v1_1/models/smplx/",
        gender="neutral",
        use_pca=True,
        num_pca_comps=6,
        flat_hand_mean=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)

    # ------------------------------------------------------------
    # 3. Converter
    # ------------------------------------------------------------
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model,
        method="pytorch"
    )

    # ------------------------------------------------------------
    # 4. Load MHR data
    # ------------------------------------------------------------
    mhr_path = "/home/haziq/datasets/mocap/data/kit/train/files_motions_292/mhr/cam1/jumping_jack01_final.npz"
    data = np.load(mhr_path, allow_pickle=True)

    verts_mhr = data["vertices"]              # [T, V, 3]
    T = verts_mhr.shape[0]
    print("num_frames:", T)

    # ------------------------------------------------------------
    # 5. Convert MHR -> SMPL-X
    # ------------------------------------------------------------
    smplx_results = converter.convert_mhr2smpl(
        mhr_vertices=verts_mhr * 100.0,
        return_smpl_meshes=True,
        return_smpl_vertices=True,
        return_smpl_parameters=True,
    )

    smplx_params = smplx_results.result_parameters

    print("\nConverter outputs:")
    for k, v in smplx_params.items():
        print(f"{k:16s}", v.shape)

    # ------------------------------------------------------------
    # 6. Move params to GPU
    #    KEEP axis-angle + PCA (do NOT convert to rotmats)
    # ------------------------------------------------------------
    for k in smplx_params:
        smplx_params[k] = to_torch(smplx_params[k], device)

    # ------------------------------------------------------------
    # 7. Explicitly provide missing pose blocks
    #    (prevents batch-size=1 defaults inside SMPL-X)
    # ------------------------------------------------------------
    smplx_params["jaw_pose"]  = torch.zeros((T, 3), device=device)
    smplx_params["leye_pose"] = torch.zeros((T, 3), device=device)
    smplx_params["reye_pose"] = torch.zeros((T, 3), device=device)

    # Optional but recommended: keep identity constant
    smplx_params["betas"] = smplx_params["betas"][0:1]  # [1,10]

    # ------------------------------------------------------------
    # 8. Forward SMPL-X
    #    (pose2rot=True by default → correct for PCA hands)
    # ------------------------------------------------------------
    out = smplx_model(**smplx_params)

    # ------------------------------------------------------------
    # 9. Render one frame
    # ------------------------------------------------------------
    faces = np.asarray(smplx_model.faces)

    frame_id = 50
    verts = out.vertices[frame_id].detach().cpu().numpy()

    _, _, image = render_simple_pyrender(
        verts,
        faces,
        "temp.png",
        img_width=800,
        img_height=800,
        ref_center=None,
        ref_radius=None,
        center_offset=(0.0, 0.0, 0.0),
        label="SMPL-X",
        save=False,
    )

    cv2.imshow("smplx", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
