"""Convert SMPL-X JSON → MHR .npz

Usage:
    pixi run python smpl_to_mhr.py \
        --smplx_json /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
        --out_npz    /home/haziq/datasets/mocap/data/fit3d/train/s03/mhr/band_pull_apart.npz

The input JSON is the dataset format produced by mhr_to_smpl.py (or fit3d-style):
    {
        "transl":           (T, 3)           float
        "global_orient":    (T, 1,  3, 3)    rotation matrices
        "body_pose":        (T, 21, 3, 3)    rotation matrices
        "betas":            (T, 10)          float
        "left_hand_pose":   (T, 15, 3, 3)    rotation matrices
        "right_hand_pose":  (T, 15, 3, 3)    rotation matrices
        "jaw_pose":         (T, 1,  3, 3)    rotation matrices
        "leye_pose":        (T, 1,  3, 3)    rotation matrices
        "reye_pose":        (T, 1,  3, 3)    rotation matrices
        "expression":       (T, 10)          float
    }

Output .npz contains:
    "vertices"  (T, V_mhr, 3)  float32  in **metres** (consistent with MHR dataset convention)
    "parameters" dict pickled as an object array (optional, via --save_params)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R

from mhr.mhr import MHR
from conversion import Conversion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_smplx_path():
    candidates = [
        os.path.expanduser("/media/haziq/Haziq/mocap/data/models_smplx_v1_1/models/smplx"),
        os.path.expanduser("~/datasets/mocap/data/models_smplx_v1_1/models/smplx"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrices [..., 3, 3] to axis-angle [..., 3]."""
    leading = rotmat.shape[:-2]
    flat = rotmat.reshape(-1, 3, 3)
    aa = R.from_matrix(flat).as_rotvec()          # (N, 3)
    return aa.reshape(*leading, 3).astype(np.float32)


def load_json_as_tensors(json_path: str, device: torch.device):
    """Load SMPL-X JSON and return a dict of float32 tensors on *device*.

    Rotation-matrix fields are converted to axis-angle automatically.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    # Fields that are stored as rotation matrices → axis-angle
    rotmat_fields = {
        "global_orient": 1,   # (T, 1,  3, 3) → (T, 3)
        "body_pose":     21,  # (T, 21, 3, 3) → (T, 63)
        "left_hand_pose": 15, # (T, 15, 3, 3) → (T, 45)
        "right_hand_pose": 15,
        "jaw_pose":      1,
        "leye_pose":     1,
        "reye_pose":     1,
    }

    params = {}
    for key, val in raw.items():
        arr = np.array(val, dtype=np.float64)

        if key in rotmat_fields:
            # arr shape might be (T, J, 3, 3) or already (T, J*3) / (T, 3)
            if arr.ndim == 4 and arr.shape[-2:] == (3, 3):
                aa = rotmat_to_aa(arr)                    # (T, J, 3)
                arr = aa.reshape(arr.shape[0], -1)        # (T, J*3)
            # else assume it's already axis-angle — pass through
        else:
            arr = arr.astype(np.float32)

        params[key] = torch.from_numpy(arr.astype(np.float32)).to(device)

    return params


def to_smplx_vertices(smplx_model, params: dict, batch_size: int = 64) -> torch.Tensor:
    """Run SMPL-X forward pass in chunks, return (T, V, 3) tensor."""
    T = params["body_pose"].shape[0]
    all_verts = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        chunk_params = {}
        for k, v in params.items():
            chunk_params[k] = v[start:end]

        # betas broadcast: if stored per-frame, use as-is; if (1,10) repeat
        if "betas" in chunk_params and chunk_params["betas"].shape[0] == 1 and (end - start) > 1:
            chunk_params["betas"] = chunk_params["betas"].expand(end - start, -1)

        with torch.no_grad():
            out = smplx_model(**chunk_params)
        all_verts.append(out.vertices.cpu())

    return torch.cat(all_verts, dim=0)  # (T, V, 3)  metres


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert SMPL-X JSON (fit3d format) to MHR .npz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smplx_json",
        type=str,
        required=True,
        help="Path to input SMPL-X JSON file.",
    )
    parser.add_argument(
        "--out_npz",
        type=str,
        default=None,
        help="Path to output MHR .npz file. Defaults to same path as input with .npz extension.",
    )
    parser.add_argument(
        "--smplx_path",
        type=str,
        default=find_smplx_path(),
        help="Path to SMPL-X model folder (contains SMPLX_NEUTRAL.pkl / .npz).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pytorch",
        choices=["pytorch", "pymomentum"],
        help="Optimization backend for MHR fitting.",
    )
    parser.add_argument(
        "--single_identity",
        action="store_true",
        default=True,
        help="Use a single shared identity (shape) across all frames.",
    )
    parser.add_argument(
        "--no_single_identity",
        dest="single_identity",
        action="store_false",
        help="Allow per-frame identity (shape) parameters.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for SMPL-X forward pass.",
    )
    parser.add_argument(
        "--save_params",
        action="store_true",
        default=False,
        help="Also save MHR parameters (rig params, betas, etc.) in the .npz.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="SMPL-X model gender.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    if args.smplx_path is None:
        sys.exit("[ERROR] SMPL-X model path not found. Pass --smplx_path explicitly.")

    if not os.path.isfile(args.smplx_json):
        sys.exit(f"[ERROR] Input JSON not found: {args.smplx_json}")

    out_npz = args.out_npz
    if out_npz is None:
        base = os.path.splitext(args.smplx_json)[0]
        out_npz = base + "_mhr.npz"

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device        : {device}")
    print(f"[INFO] smplx_json    : {args.smplx_json}")
    print(f"[INFO] out_npz       : {out_npz}")
    print(f"[INFO] method        : {args.method}")
    print(f"[INFO] single_id     : {args.single_identity}")

    # ------------------------------------------------------------------
    # 1) Load JSON → axis-angle tensors
    # ------------------------------------------------------------------
    print("\n[1/4] Loading SMPL-X JSON ...")
    params = load_json_as_tensors(args.smplx_json, device)
    T_full = params["body_pose"].shape[0]
    print(f"      total frames : {T_full}")
    for k, v in params.items():
        print(f"      {k:20s} {tuple(v.shape)}")

    # ------------------------------------------------------------------
    # 2) Identify valid frames (no NaN/Inf)
    # ------------------------------------------------------------------
    valid_mask = torch.ones(T_full, dtype=torch.bool, device=device)
    for v in params.values():
        valid_mask &= torch.isfinite(v).all(dim=tuple(range(1, v.ndim)))

    good_idx = torch.where(valid_mask)[0].cpu().numpy()
    bad_idx  = torch.where(~valid_mask)[0].cpu().numpy()
    print(f"\n[INFO] good={len(good_idx)}  bad={len(bad_idx)}")
    if len(bad_idx):
        print(f"[WARN] bad frames (first 20): {bad_idx[:20].tolist()}")

    if len(good_idx) == 0:
        sys.exit("[ERROR] All frames contain NaN/Inf — nothing to process.")

    params_good = {k: v[good_idx] for k, v in params.items()}
    T = len(good_idx)

    # ------------------------------------------------------------------
    # 3) SMPL-X forward pass → vertices (metres)
    # ------------------------------------------------------------------
    print(f"\n[2/4] Running SMPL-X forward pass ({T} frames) ...")
    smplx_model = smplx.SMPLX(
        model_path=args.smplx_path,
        gender=args.gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)

    smplx_verts = to_smplx_vertices(smplx_model, params_good, batch_size=args.batch_size)
    # smplx_verts: (T, 10475, 3)  in metres
    print(f"      SMPL-X vertices shape : {tuple(smplx_verts.shape)}")

    # ------------------------------------------------------------------
    # 4) Convert SMPL-X vertices → MHR
    # ------------------------------------------------------------------
    print(f"\n[3/4] Converting to MHR (method={args.method}) ...")
    mhr_model = MHR.from_files(lod=1, device=device)
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model,
        method=args.method,
    )

    # Conversion class expects vertices in metres; internally converts to cm.
    conversion_result = converter.convert_smpl2mhr(
        smpl_vertices=smplx_verts.to(device),
        smpl_parameters=None,
        single_identity=args.single_identity,
        return_mhr_meshes=False,
        return_mhr_vertices=True,
        return_mhr_parameters=args.save_params,
        return_fitting_errors=True,
    )

    mhr_verts_cm = conversion_result.result_vertices  # (T, V_mhr, 3) in cm
    if isinstance(mhr_verts_cm, torch.Tensor):
        mhr_verts_cm = mhr_verts_cm.detach().cpu().numpy()
    elif isinstance(mhr_verts_cm, list):
        mhr_verts_cm = np.stack([
            v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            for v in mhr_verts_cm
        ], axis=0)

    mhr_verts_m = (mhr_verts_cm / 100.0).astype(np.float32)  # → metres

    print(f"      MHR vertices shape : {mhr_verts_m.shape}")
    if conversion_result.result_errors is not None:
        errs = conversion_result.result_errors
        if hasattr(errs, "__len__") and len(errs):
            errs_np = np.asarray(errs)
            print(f"      Fitting error (cm) — mean={errs_np.mean():.4f}  max={errs_np.max():.4f}")

    # ------------------------------------------------------------------
    # 5) Scatter back to T_full with NaNs for bad frames
    # ------------------------------------------------------------------
    full_verts = np.full((T_full, *mhr_verts_m.shape[1:]), np.nan, dtype=np.float32)
    full_verts[good_idx] = mhr_verts_m

    # ------------------------------------------------------------------
    # 6) Save .npz
    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving → {out_npz}")
    save_dict = {"vertices": full_verts}

    if args.save_params and conversion_result.result_parameters is not None:
        mhr_params = conversion_result.result_parameters
        for k, v in mhr_params.items():
            arr = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            save_dict[f"param_{k}"] = arr.astype(np.float32)

    np.savez_compressed(out_npz, **save_dict)
    print(f"\n✅ Done. Saved {T_full} frames ({len(good_idx)} valid) to: {out_npz}")
    print(f"   vertices  : {full_verts.shape}  dtype={full_verts.dtype}  (metres)")


if __name__ == "__main__":
    main()
