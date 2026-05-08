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
# Triptych video renderer
# ---------------------------------------------------------------------------

def _render_triptych_video(full_verts, mhr_faces, out_video, fps=30.0, panel_h=360):
    """
    Render a 3-panel side-by-side video of the MHR mesh sequence.

    Each frame shows the same MHR mesh three times, evenly spaced horizontally,
    on a white background — matching the style of mhr/{action}.mp4 in the
    fit3d dataset.

    full_verts : (T, V, 3) float32, metres. NaN rows → blank white frame.
    mhr_faces  : (F, 3) int32
    out_video  : output .mp4 path
    fps        : frames per second
    panel_h    : height in pixels; total output is (3*panel_h) × panel_h
    """
    import cv2
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    T, V, _ = full_verts.shape
    panel_w = panel_h * 3   # 3:1 aspect (3 square panels side by side)

    # ── reference frame for stable camera ────────────────────────────────────
    valid_mask = np.isfinite(full_verts).all(axis=(1, 2))
    if not valid_mask.any():
        print("[VIDEO] No valid frames — skipping video render.")
        return

    ref_verts = full_verts[np.argmax(valid_mask)]   # (V, 3)
    vmin = ref_verts.min(axis=0)
    vmax = ref_verts.max(axis=0)
    center = (vmin + vmax) * 0.5
    mesh_w = float(vmax[0] - vmin[0])
    gap    = mesh_w + 0.3   # horizontal spacing between mesh centres

    # ── camera: covers all 3 meshes ──────────────────────────────────────────
    combined_half_w = gap + mesh_w * 0.5
    combined_diag   = np.sqrt(
        (2 * combined_half_w) ** 2
        + (vmax[1] - vmin[1]) ** 2
        + (vmax[2] - vmin[2]) ** 2
    )
    ref_radius = float(combined_diag) * 0.5
    cam_dist   = 2.5 * ref_radius

    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, 3] = center + np.array([0.0, 0.0, cam_dist], dtype=np.float32)

    scene = pyrender.Scene(
        ambient_light=[0.4, 0.4, 0.4],
        bg_color=[1.0, 1.0, 1.0],
    )
    camera   = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=3.0)
    cam_node = scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=cam_pose)

    renderer   = pyrender.OffscreenRenderer(panel_w, panel_h)
    x_offsets  = np.array([-gap, 0.0, gap], dtype=np.float32)
    mesh_nodes = [None, None, None]

    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (panel_w, panel_h))

    blank = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    vc    = np.full((V, 4), 0.8, dtype=np.float32)   # grey vertex colour
    vc[:, 3] = 1.0

    print(f"\n[VIDEO] Rendering {T} frames → {out_video}")
    print(f"  output : {panel_w}×{panel_h} px   fps={fps:.1f}   gap={gap:.3f} m")

    for t in range(T):
        if t % 100 == 0:
            print(f"  [{t}/{T}]")

        verts = full_verts[t]
        if not np.isfinite(verts).all():
            writer.write(blank)
            continue

        # Remove previous mesh nodes
        for i in range(3):
            if mesh_nodes[i] is not None:
                scene.remove_node(mesh_nodes[i])
                mesh_nodes[i] = None

        # Add 3 copies with X offset
        for i, dx in enumerate(x_offsets):
            v = verts.copy()
            v[:, 0] += dx
            tri     = trimesh.Trimesh(vertices=v, faces=mhr_faces,
                                      vertex_colors=vc, process=False)
            mesh_pr = pyrender.Mesh.from_trimesh(tri, smooth=True)
            mesh_nodes[i] = scene.add(mesh_pr)

        color, _ = renderer.render(scene)
        writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    writer.release()
    renderer.delete()
    print(f"✅ Video saved → {out_video}")


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
    # ── Video rendering ──────────────────────────────────────────────────────
    parser.add_argument(
        "--out_video",
        type=str,
        default=None,
        help="Path to output triptych .mp4 (default: same as --out_npz with .mp4 extension).",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        default=False,
        help="Skip triptych video rendering.",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=30.0,
        help="Output video frame rate.",
    )
    parser.add_argument(
        "--video_panel_h",
        type=int,
        default=360,
        help="Height of each panel in pixels (total width = 3 × panel_h).",
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

    out_video = args.out_video
    if out_video is None and not args.no_video:
        out_video = os.path.splitext(out_npz)[0] + ".mp4"

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device        : {device}")
    print(f"[INFO] smplx_json    : {args.smplx_json}")
    print(f"[INFO] out_npz       : {out_npz}")
    print(f"[INFO] out_video     : {out_video if out_video else '(skipped)'}")
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

    # ── Optional: render triptych video ──────────────────────────────────────
    if out_video is not None:
        mhr_faces = np.asarray(mhr_model.character.mesh.faces, dtype=np.int32)
        _render_triptych_video(
            full_verts,
            mhr_faces,
            out_video,
            fps=args.video_fps,
            panel_h=args.video_panel_h,
        )


if __name__ == "__main__":
    main()
