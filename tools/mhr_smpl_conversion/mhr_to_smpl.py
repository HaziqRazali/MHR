import os
import sys
import json
import cv2
import torch
import smplx
import argparse
import numpy as np

from mhr.mhr import MHR
from conversion import Conversion
from smplx.lbs import batch_rodrigues

sys.path.append(os.path.expanduser('~/datasets/mocap/my_scripts/'))
from utils_draw import render_simple_pyrender


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def to_torch(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")
    return x.to(device=device, dtype=torch.float32)


def describe(x):
    if isinstance(x, torch.Tensor):
        return f"torch.Tensor {tuple(x.shape)} {x.dtype}"
    if isinstance(x, np.ndarray):
        return f"np.ndarray {x.shape} {x.dtype}"
    if isinstance(x, list):
        if len(x) == 0:
            return "list (empty)"
        return f"list (len={len(x)}), elem type={type(x[0]).__name__}"
    return type(x).__name__


def aa_to_rotmat(aa):
    """
    aa: [..., 3] axis-angle
    -> [..., 3, 3] rotation matrices
    """
    return batch_rodrigues(aa.reshape(-1, 3)).reshape(*aa.shape[:-1], 3, 3)


def expand_hand_pca_to_aa(smplx_model_pca, hand_pca, is_left=True):
    """
    Convert PCA hand pose -> full 15-joint axis-angle.

    hand_pca: [T, num_pca_comps]  (e.g. [T,6])
    return:   [T, 45] axis-angle  (15 joints * 3)
    """
    if is_left:
        comps = smplx_model_pca.left_hand_components
        mean  = smplx_model_pca.left_hand_mean
    else:
        comps = smplx_model_pca.right_hand_components
        mean  = smplx_model_pca.right_hand_mean

    comps = comps.to(hand_pca.device, hand_pca.dtype)
    mean  = mean.to(hand_pca.device, hand_pca.dtype)

    # Robust to either [num_comps,45] or [45,num_comps]
    if comps.shape[0] == hand_pca.shape[1]:
        aa45 = hand_pca @ comps
    else:
        aa45 = hand_pca @ comps.T

    return aa45 + mean.unsqueeze(0)  # [T,45]


def to_list(x):
    """torch tensor -> JSON-safe python nested lists"""
    return x.detach().cpu().numpy().tolist()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

# python mhr_to_smpl.py --mhr_path /home/haziq/datasets/mocap/data/kit/train/files_motions_292/mhr/cam1/jumping_jack01_final.npz --out_json /home/haziq/datasets/mocap/data/kit/train/files_motions_292/smplx/cam1/jumping_jack01_final.json --show 1 --frame_id 50

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mhr_path",
        type=str,
        required=True,
        help="Path to input .npz MHR file (must contain 'vertices').",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Path to output .json to save SMPL-X params.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="If 1, print format + show OpenCV render preview. Default 0.",
    )
    parser.add_argument(
        "--frame_id",
        type=int,
        default=50,
        help="Frame index for visualization when --show=1. Default 50.",
    )
    parser.add_argument(
        "--smplx_path",
        type=str,
        default="~/datasets/mocap/data/models_smplx_v1_1/models/smplx/",
        help="Path to SMPL-X model folder (contains SMPLX_*.pkl). Default is your usual path.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scale applied to MHR vertices before conversion (keeps your previous behavior).",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.show:
        print("Using device:", device)

    SMPLX_PATH = os.path.expanduser(args.smplx_path)

    if not os.path.isdir(SMPLX_PATH):
        raise FileNotFoundError(f"SMPL-X model_path does not exist: {SMPLX_PATH}")

    if not os.path.isfile(args.mhr_path):
        raise FileNotFoundError(f"Input mhr_path does not exist: {args.mhr_path}")

    # ------------------------------------------------------------
    # 1) Models
    # ------------------------------------------------------------
    mhr_model = MHR.from_files(lod=1, device=device)

    # PCA model for conversion (converter returns [T,6] hand PCA)
    smplx_model_pca = smplx.SMPLX(
        model_path=SMPLX_PATH,
        gender="neutral",
        use_pca=True,
        num_pca_comps=6,
        flat_hand_mean=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)

    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model_pca,
        method="pytorch"
    )

    # ------------------------------------------------------------
    # 2) Load MHR
    # ------------------------------------------------------------
    data = np.load(args.mhr_path, allow_pickle=True)
    if "vertices" not in data.files:
        raise KeyError(f"Expected key 'vertices' in {args.mhr_path}, got keys: {data.files}")

    verts_mhr = data["vertices"]
    if verts_mhr.ndim != 3 or verts_mhr.shape[-1] != 3:
        raise ValueError(f"Expected vertices shape [T,V,3], got {verts_mhr.shape}")

    T = verts_mhr.shape[0]
    if args.show:
        print("num_frames:", T)

    # ------------------------------------------------------------
    # 3) Convert MHR -> SMPL-X params (PCA hands)
    # ------------------------------------------------------------
    smplx_results = converter.convert_mhr2smpl(
        mhr_vertices=verts_mhr * float(args.scale),
        return_smpl_parameters=True,
    )

    smplx_params = smplx_results.result_parameters
    for k in smplx_params:
        smplx_params[k] = to_torch(smplx_params[k], device)

    if args.show:
        print("\nConverter outputs:")
        for k, v in smplx_params.items():
            if torch.is_tensor(v):
                print(f"{k:16s} {tuple(v.shape)}")
            else:
                print(f"{k:16s} {type(v)}")

    # ------------------------------------------------------------
    # 4) Expand hands: PCA (6) -> FULL (45)
    # ------------------------------------------------------------
    if "left_hand_pose" not in smplx_params or "right_hand_pose" not in smplx_params:
        raise KeyError("Expected left_hand_pose and right_hand_pose in converter outputs.")

    smplx_params["left_hand_pose"] = expand_hand_pca_to_aa(
        smplx_model_pca, smplx_params["left_hand_pose"], is_left=True
    )
    smplx_params["right_hand_pose"] = expand_hand_pca_to_aa(
        smplx_model_pca, smplx_params["right_hand_pose"], is_left=False
    )

    # ------------------------------------------------------------
    # 5) Add missing pose blocks + stabilize betas
    # ------------------------------------------------------------
    smplx_params["jaw_pose"]  = torch.zeros((T, 3), device=device)
    smplx_params["leye_pose"] = torch.zeros((T, 3), device=device)
    smplx_params["reye_pose"] = torch.zeros((T, 3), device=device)

    if "betas" in smplx_params and smplx_params["betas"].ndim == 2 and smplx_params["betas"].shape[0] == T:
        smplx_params["betas"] = smplx_params["betas"][0:1]  # [1,10]

    # ------------------------------------------------------------
    # 6) SMPL-X model (NO PCA, FULL HANDS)
    # ------------------------------------------------------------
    smplx_model = smplx.SMPLX(
        model_path=SMPLX_PATH,
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)

    # ------------------------------------------------------------
    # 7) Forward pass (uses 15-joint hands via [T,45])
    # ------------------------------------------------------------
    out = smplx_model(**smplx_params)

    # ------------------------------------------------------------
    # 8) Build JSON (dataset-compatible format)
    # ------------------------------------------------------------
    global_R = aa_to_rotmat(smplx_params["global_orient"]).view(T, 1, 3, 3)
    body_R   = aa_to_rotmat(smplx_params["body_pose"].view(T, 21, 3))
    left_R   = aa_to_rotmat(smplx_params["left_hand_pose"].view(T, 15, 3))
    right_R  = aa_to_rotmat(smplx_params["right_hand_pose"].view(T, 15, 3))
    jaw_R    = aa_to_rotmat(smplx_params["jaw_pose"]).view(T, 1, 3, 3)
    leye_R   = aa_to_rotmat(smplx_params["leye_pose"]).view(T, 1, 3, 3)
    reye_R   = aa_to_rotmat(smplx_params["reye_pose"]).view(T, 1, 3, 3)

    # betas: store per-frame (10,) to match your dataset print format
    if "betas" in smplx_params:
        betas = smplx_params["betas"]
        if betas.ndim == 2 and betas.shape[0] == 1:
            betas_T = betas.repeat(T, 1)
        elif betas.ndim == 2 and betas.shape[0] == T:
            betas_T = betas
        elif betas.ndim == 1 and betas.shape[0] == 10:
            betas_T = betas.view(1, 10).repeat(T, 1)
        else:
            raise ValueError(f"Unexpected betas shape: {tuple(betas.shape)}")
    else:
        betas_T = torch.zeros((T, 10), device=device)

    if "expression" in smplx_params:
        expr = smplx_params["expression"]
        if expr.ndim == 2 and expr.shape[0] == T:
            expr_T = expr
        elif expr.ndim == 2 and expr.shape[0] == 1:
            expr_T = expr.repeat(T, 1)
        else:
            expr_T = torch.zeros((T, 10), device=device)
    else:
        expr_T = torch.zeros((T, 10), device=device)

    json_data = {
        "transl":          to_list(smplx_params["transl"]),
        "global_orient":   to_list(global_R),
        "body_pose":       to_list(body_R),
        "betas":           to_list(betas_T),
        "left_hand_pose":  to_list(left_R),
        "right_hand_pose": to_list(right_R),
        "jaw_pose":        to_list(jaw_R),
        "leye_pose":       to_list(leye_R),
        "reye_pose":       to_list(reye_R),
        "expression":      to_list(expr_T),
    }

    # ------------------------------------------------------------
    # 9) Optional show: print format + render
    # ------------------------------------------------------------
    if args.show:
        print("\nSMPLX format")
        print("#####")
        for k, v in json_data.items():
            print(f"{k:20s}: {describe(v)}")
        print()
        for k, v in json_data.items():
            arr = np.asarray(v[0])
            print(f"{k:20s} per-frame shape: {arr.shape}")
        print("#####")

    # ------------------------------------------------------------
    # 10) Save JSON
    # ------------------------------------------------------------
    ensure_parent_dir(args.out_json)
    with open(args.out_json, "w") as f:
        json.dump(json_data, f)
    if args.show:
        print(f"\n✅ Saved JSON to: {args.out_json}")

    # ------------------------------------------------------------
    # 11) Optional render sanity check
    # ------------------------------------------------------------
    if args.show:
        faces = np.asarray(smplx_model.faces)
        frame_id = int(args.frame_id)
        frame_id = max(0, min(frame_id, T - 1))

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
            label="SMPL-X FULL HAND",
            save=False,
        )
        cv2.imshow("smplx", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
