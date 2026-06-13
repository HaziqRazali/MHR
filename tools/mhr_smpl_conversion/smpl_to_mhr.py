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
        os.path.expanduser("/data/haziq/mocap/data/models_smplx_v1_1/models/smplx"),
        os.path.expanduser("/data/haziq/mocap/data/models_smplx_v1_1/models/smplx"),
        os.path.expanduser("/data/mocap/data/models_smplx_v1_1/models/smplx"),
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


def load_npz_as_tensors(npz_path: str, device: torch.device):
    """Load a synthesizer-format SMPL-X NPZ and return a dict of float32 tensors.

    The NPZ contains rotation matrices (same as the JSON format) stored as
    numpy arrays.  Converts them to axis-angle to match load_json_as_tensors.
    """
    data = np.load(npz_path, allow_pickle=True)

    # Fields stored as rotation matrices → axis-angle
    rotmat_fields = {
        "global_orient": 1,   # (T, 1,  3, 3) → (T, 3)   or (T, 3, 3) → (T, 3)
        "body_pose":     21,  # (T, 21, 3, 3) → (T, 63)
    }

    params = {}
    for key, arr in data.items():
        if not isinstance(arr, np.ndarray):
            continue
        arr = arr.astype(np.float64)

        if key in rotmat_fields:
            # Handle (T, 3, 3) → add joint dim
            if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
                arr = arr[:, np.newaxis, :, :]  # (T, 1, 3, 3)
            if arr.ndim == 4 and arr.shape[-2:] == (3, 3):
                aa = rotmat_to_aa(arr)                    # (T, J, 3)
                arr = aa.reshape(arr.shape[0], -1)        # (T, J*3)
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
# Input-video auto-detection
# ---------------------------------------------------------------------------

def find_input_video(smplx_json: str, camera_id: str = None):
    """
    Given the SMPL-X JSON path:
        $DATA_ROOT/{split}/{subj}/smplx/<action>.json
    return the corresponding original video:
        $DATA_ROOT/{split}/{subj}/videos/{camera_id}/<action>.mp4

    If *camera_id* is None the first camera subfolder (sorted) is used.
    Returns None if nothing is found.
    """
    smplx_dir  = os.path.dirname(smplx_json)        # .../smplx
    subj_dir   = os.path.dirname(smplx_dir)          # .../s03
    videos_dir = os.path.join(subj_dir, "videos")
    action     = os.path.splitext(os.path.basename(smplx_json))[0]

    if not os.path.isdir(videos_dir):
        return None

    if camera_id is not None:
        candidate = os.path.join(videos_dir, camera_id, f"{action}.mp4")
        return candidate if os.path.isfile(candidate) else None

    try:
        cam_dirs = sorted(
            d for d in os.listdir(videos_dir)
            if os.path.isdir(os.path.join(videos_dir, d))
        )
    except OSError:
        return None

    for cam in cam_dirs:
        candidate = os.path.join(videos_dir, cam, f"{action}.mp4")
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Comparison video renderer  (video | MHR | SMPL-X)
# ---------------------------------------------------------------------------

def _render_comparison_video(
    mhr_verts,
    mhr_faces,
    smplx_verts,
    smplx_faces,
    out_video,
    input_video=None,
    fps=30.0,
    panel_h=360,
):
    """
    Render a 3-panel side-by-side comparison video:
        [ original video | MHR mesh | SMPL-X mesh ]

    A label bar with 'mhr' / 'smpl' is drawn below each mesh panel.

    mhr_verts   : (T, V_mhr,   3) float32, metres. NaN rows → white panel.
    mhr_faces   : (F_mhr,  3) int32
    smplx_verts : (T, V_smplx, 3) float32, metres. NaN rows → white panel.
    smplx_faces : (F_smplx, 3) int32
    out_video   : output .mp4 path
    input_video : path to the original camera video (optional)
    fps         : frames per second
    panel_h     : height of each square panel in pixels
    """
    import cv2
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    T       = mhr_verts.shape[0]
    label_h = 30                       # pixel height of the text label bar
    panel_total_h = panel_h + label_h
    out_w   = panel_h * 3              # three square panels wide

    # ── Stable camera: bounding box of first valid MHR frame ─────────────────
    valid_mhr = np.isfinite(mhr_verts).all(axis=(1, 2))
    if valid_mhr.any():
        ref_verts = mhr_verts[np.argmax(valid_mhr)]
    else:
        valid_s = np.isfinite(smplx_verts).all(axis=(1, 2))
        if not valid_s.any():
            print("[VIDEO] No valid frames — skipping video render.")
            return
        ref_verts = smplx_verts[np.argmax(valid_s)]

    vmin   = ref_verts.min(axis=0)
    vmax   = ref_verts.max(axis=0)
    center = (vmin + vmax) * 0.5
    diag   = float(np.linalg.norm(vmax - vmin))

    # Visualizer-only rotation — the saved .npz is unaffected.
    # MHR native space: Z-up, Y-depth.  Camera sits at +Z looking toward -Z,
    # so without correction we see a top-down blob.
    # Empirically determined: apply Rx=270° then Ry=270° to get the character
    # standing upright and facing the camera.
    #   R = Ry(270°) @ Rx(270°)
    _Rx270 = np.array([[1.,  0.,  0.],
                       [0.,  0.,  1.],
                       [0., -1.,  0.]], dtype=np.float32)
    _Ry270 = np.array([[ 0.,  0., -1.],
                       [ 0.,  1.,  0.],
                       [ 1.,  0.,  0.]], dtype=np.float32)
    viz_rot = _Ry270 @ _Rx270   # [[0,1,0],[0,0,1],[1,0,0]]

    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, 3] = center + np.array([0.0, 0.0, diag * 1.1], dtype=np.float32)

    # ── Two persistent pyrender scenes (one per mesh type) ───────────────────
    def _make_scene():
        s = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[1.0, 1.0, 1.0])
        s.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0), pose=cam_pose)
        s.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=cam_pose)
        return s

    scene_mhr   = _make_scene()
    scene_smplx = _make_scene()
    renderer    = pyrender.OffscreenRenderer(panel_h, panel_h)

    blank_rgb = np.full((panel_h, panel_h, 3), 255, dtype=np.uint8)

    vc_mhr   = np.full((mhr_verts.shape[1],   4), [0.75, 0.82, 0.95, 1.0], dtype=np.float32)
    vc_smplx = np.full((smplx_verts.shape[1], 4), [0.95, 0.80, 0.70, 1.0], dtype=np.float32)

    # ── Video capture ─────────────────────────────────────────────────────────
    vcap = None
    if input_video and os.path.isfile(input_video):
        vcap = cv2.VideoCapture(input_video)
        if not vcap.isOpened():
            vcap = None
            print(f"[WARN] Cannot open video: {input_video}")
        else:
            print(f"[VIDEO] Input video    : {input_video}")

    # ── Output writer ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (out_w, panel_total_h))

    print(f"\n[VIDEO] Rendering {T} frames → {out_video}")
    print(f"        output : {out_w}×{panel_total_h} px   fps={fps:.1f}")

    node_mhr   = None
    node_smplx = None

    def _make_label_bar(text):
        bar = np.full((label_h, panel_h, 3), 230, dtype=np.uint8)
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness  = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = max(0, (panel_h - tw) // 2)
        y = (label_h + th) // 2
        cv2.putText(bar, text, (x, y), font, font_scale, (40, 40, 40), thickness, cv2.LINE_AA)
        return bar

    label_mhr   = _make_label_bar("mhr")
    label_smplx = _make_label_bar("smpl")
    label_blank = np.full((label_h, panel_h, 3), 230, dtype=np.uint8)

    for t in range(T):
        if t % 100 == 0:
            print(f"  [{t}/{T}]")

        # -- MHR panel --------------------------------------------------------
        if node_mhr is not None:
            scene_mhr.remove_node(node_mhr)
            node_mhr = None
        vt = mhr_verts[t]
        if np.isfinite(vt).all():
            vt = (vt - center) @ viz_rot.T + center
            tri      = trimesh.Trimesh(vertices=vt, faces=mhr_faces,
                                       vertex_colors=vc_mhr, process=False)
            node_mhr = scene_mhr.add(pyrender.Mesh.from_trimesh(tri, smooth=True))
            color, _ = renderer.render(scene_mhr)
            panel_mhr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        else:
            panel_mhr = blank_rgb.copy()

        # -- SMPL-X panel -----------------------------------------------------
        if node_smplx is not None:
            scene_smplx.remove_node(node_smplx)
            node_smplx = None
        vs = smplx_verts[t]
        if np.isfinite(vs).all():
            vs = (vs - center) @ viz_rot.T + center
            tri        = trimesh.Trimesh(vertices=vs, faces=smplx_faces,
                                         vertex_colors=vc_smplx, process=False)
            node_smplx = scene_smplx.add(pyrender.Mesh.from_trimesh(tri, smooth=True))
            color, _   = renderer.render(scene_smplx)
            panel_smplx = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        else:
            panel_smplx = blank_rgb.copy()

        # -- Video panel ------------------------------------------------------
        if vcap is not None:
            vcap.set(cv2.CAP_PROP_POS_FRAMES, float(t))
            ret, frame = vcap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                # Scale to panel_h height, then center-crop width to panel_h
                if h != panel_h:
                    new_w  = max(1, int(round(w * panel_h / h)))
                    frame  = cv2.resize(frame, (new_w, panel_h), interpolation=cv2.INTER_AREA)
                h2, w2 = frame.shape[:2]
                if w2 > panel_h:
                    x0    = (w2 - panel_h) // 2
                    frame = frame[:, x0:x0 + panel_h]
                elif w2 < panel_h:
                    pad = np.full((panel_h, panel_h, 3), 200, dtype=np.uint8)
                    pad[:, (panel_h - w2) // 2:(panel_h - w2) // 2 + w2] = frame
                    frame = pad
                panel_vid = frame
            else:
                panel_vid = np.full((panel_h, panel_h, 3), 200, dtype=np.uint8)
        else:
            panel_vid = np.full((panel_h, panel_h, 3), 200, dtype=np.uint8)

        # -- Compose ----------------------------------------------------------
        vid_full   = np.vstack([panel_vid,   label_blank])
        mhr_full   = np.vstack([panel_mhr,   label_mhr])
        smplx_full = np.vstack([panel_smplx, label_smplx])
        writer.write(np.hstack([vid_full, mhr_full, smplx_full]))

    writer.release()
    renderer.delete()
    if vcap is not None:
        vcap.release()
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
        "--video_only",
        action="store_true",
        default=False,
        help=(
            "Skip SMPL-X forward pass and MHR conversion. "
            "Load an existing --out_npz and render the video only. "
            "--smplx_json is optional (used solely for input-video auto-detection)."
        ),
    )
    parser.add_argument(
        "--smplx_json",
        type=str,
        default=None,
        help="Path to input SMPL-X JSON file. Required unless --video_only is set.",
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
    parser.add_argument(
        "--input_video",
        type=str,
        default=None,
        help=(
            "Path to the original camera video shown on the left panel. "
            "Auto-detected from <subj>/videos/<camera_id>/<action>.mp4 if not given."
        ),
    )
    parser.add_argument(
        "--camera_id",
        type=str,
        default=None,
        help=(
            "Camera sub-folder name to use when auto-detecting the input video "
            "(e.g. '50591643'). If not set, the first sorted camera folder is used."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    if not args.video_only:
        if args.smplx_path is None:
            sys.exit("[ERROR] SMPL-X model path not found. Pass --smplx_path explicitly.")
        if args.smplx_json is None:
            sys.exit("[ERROR] --smplx_json is required unless --video_only is set.")
        if not os.path.isfile(args.smplx_json):
            sys.exit(f"[ERROR] Input JSON not found: {args.smplx_json}")

    out_npz = args.out_npz
    if out_npz is None:
        if args.smplx_json:
            base = os.path.splitext(args.smplx_json)[0]
            out_npz = base + "_mhr.npz"
        else:
            sys.exit("[ERROR] --out_npz is required when --smplx_json is not provided.")

    out_video = args.out_video
    if out_video is None and not args.no_video:
        out_video = os.path.splitext(out_npz)[0] + ".mp4"

    # Auto-detect input video (original camera footage for left panel).
    # In --video_only mode we derive the subject dir from the npz path
    # (.../subj/mhr/action.npz) and construct a fake smplx_json path so
    # find_input_video can locate .../subj/videos/<cam>/<action>.mp4.
    input_video = args.input_video
    if input_video is None and out_video is not None:
        _json_for_vid = args.smplx_json
        if _json_for_vid is None and args.video_only:
            _action   = os.path.splitext(os.path.basename(out_npz))[0]
            _subj_dir = os.path.dirname(os.path.dirname(out_npz))  # .../subj
            _json_for_vid = os.path.join(_subj_dir, "smplx", _action + ".json")
        if _json_for_vid is not None:
            input_video = find_input_video(_json_for_vid, camera_id=args.camera_id)
        if input_video:
            print(f"[INFO] input_video   : {input_video} (auto-detected)")
        else:
            print("[INFO] input_video   : (not found — left panel will be grey)")

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device        : {device}")
    if args.smplx_json:
        print(f"[INFO] smplx_json    : {args.smplx_json}")
    print(f"[INFO] out_npz       : {out_npz}")
    print(f"[INFO] out_video     : {out_video if out_video else '(skipped)'}")
    if not args.video_only:
        print(f"[INFO] method        : {args.method}")
        print(f"[INFO] single_id     : {args.single_identity}")

    # ------------------------------------------------------------------
    # --video_only: load existing .npz, run MHR forward pass, render
    # ------------------------------------------------------------------
    if args.video_only:
        if not os.path.isfile(out_npz):
            sys.exit(f"[ERROR] --video_only requires an existing .npz: {out_npz}")
        if out_video is None:
            sys.exit("[ERROR] --video_only with --no_video does nothing. Aborting.")

        print(f"\n[video_only] Loading .npz → {out_npz}")
        data       = np.load(out_npz)
        T_full     = data["body_pose_params"].shape[0]

        valid_frame_mask = np.isfinite(data["body_pose_params"]).all(axis=1)

        # Reconstruct lbs_model_params (204-dim): trans | orient | body | scale(=0)
        lbs_full  = np.zeros((T_full, 204), dtype=np.float32)
        lbs_full[:, 0:3]   = data["global_trans"]
        lbs_full[:, 3:6]   = data["global_orient"]
        lbs_full[:, 6:136] = data["body_pose_params"]
        id_full   = data["shape_params"].astype(np.float32)   # (T_full, S)
        expr_full = data["expr_params"].astype(np.float32)    # (T_full, F)

        print(f"       frames : {T_full}  valid : {valid_frame_mask.sum()}")

        mhr_model = MHR.from_files(lod=1, device=device)
        mhr_faces = np.asarray(mhr_model.character.mesh.faces, dtype=np.int32)

        print(f"[video_only] Running MHR forward pass ({T_full} frames) ...")
        recon_list = []
        for start in range(0, T_full, args.batch_size):
            end = min(start + args.batch_size, T_full)
            with torch.no_grad():
                v, _ = mhr_model(
                    identity_coeffs  = torch.from_numpy(id_full[start:end]).to(device),
                    model_parameters = torch.from_numpy(lbs_full[start:end]).to(device),
                    face_expr_coeffs = torch.from_numpy(expr_full[start:end]).to(device),
                )
            recon_list.append(v.cpu().numpy())
        recon_verts = np.concatenate(recon_list, axis=0) / 100.0  # cm → m
        recon_verts[~valid_frame_mask] = np.nan

        # Run SMPL-X forward pass from the JSON if available (cheap — just LBS)
        smplx_verts_render = None
        smplx_faces_render = None
        if _json_for_vid is not None and os.path.isfile(_json_for_vid) and args.smplx_path is not None:
            print(f"[video_only] Running SMPL-X forward pass from {_json_for_vid} ...")
            _params = load_json_as_tensors(_json_for_vid, device)
            _T_json = _params["body_pose"].shape[0]
            _valid  = torch.ones(_T_json, dtype=torch.bool, device=device)
            for _v in _params.values():
                _valid &= torch.isfinite(_v).all(dim=tuple(range(1, _v.ndim)))
            _good_idx = torch.where(_valid)[0].cpu().numpy()
            _params_good = {k: v[_good_idx] for k, v in _params.items()}
            _smplx_model = smplx.SMPLX(
                model_path=args.smplx_path,
                gender=args.gender,
                use_pca=False,
                flat_hand_mean=True,
                num_betas=10,
                num_expression_coeffs=10,
            ).to(device)
            _sv_np = to_smplx_vertices(_smplx_model, _params_good, batch_size=args.batch_size).cpu().numpy()
            _V = _sv_np.shape[1]
            smplx_verts_render = np.full((_T_json, _V, 3), np.nan, dtype=np.float32)
            smplx_verts_render[_good_idx] = _sv_np
            # Align frame count to T_full (trim or pad with NaN if lengths differ)
            if _T_json != T_full:
                print(f"[WARN] JSON has {_T_json} frames but .npz has {T_full} — aligning")
                _T_min = min(_T_json, T_full)
                _aligned = np.full((T_full, _V, 3), np.nan, dtype=np.float32)
                _aligned[:_T_min] = smplx_verts_render[:_T_min]
                smplx_verts_render = _aligned
            smplx_faces_render = _smplx_model.faces.astype(np.int32)
        else:
            print("[video_only] SMPL-X JSON not found — SMPL-X panel will be blank.")
            smplx_verts_render = np.full((T_full, 1, 3), np.nan, dtype=np.float32)
            smplx_faces_render = np.zeros((1, 3), dtype=np.int32)

        _render_comparison_video(
            mhr_verts=recon_verts,
            mhr_faces=mhr_faces,
            smplx_verts=smplx_verts_render,
            smplx_faces=smplx_faces_render,
            out_video=out_video,
            input_video=input_video,
            fps=args.video_fps,
            panel_h=args.video_panel_h,
        )
        return

    # ------------------------------------------------------------------
    # 1) Load input → axis-angle tensors  (.json or .npz auto-detected)
    # ------------------------------------------------------------------
    _ext = os.path.splitext(args.smplx_json)[1].lower()
    if _ext == ".npz":
        print("\n[1/4] Loading SMPL-X NPZ ...")
        params = load_npz_as_tensors(args.smplx_json, device)
    else:
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
    smplx_faces = smplx_model.faces.astype(np.int32)   # (F, 3) — needed for video

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
        is_tracking=True,             # warm-start each frame from previous (better for sequences)
        return_mhr_meshes=False,
        return_mhr_vertices=True,
        return_mhr_parameters=True,   # always needed to save named params
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
    # 5) Extract named MHR parameters & scatter back to T_full
    # ------------------------------------------------------------------
    fit_params  = conversion_result.result_parameters
    lbs_np   = fit_params["lbs_model_params"].detach().cpu().numpy().astype(np.float32)  # (T, 204)
    id_np    = fit_params["identity_coeffs"].detach().cpu().numpy().astype(np.float32)   # (T, S)
    expr_np  = fit_params["face_expr_coeffs"].detach().cpu().numpy().astype(np.float32)  # (T, F)

    # MHR lbs_model_params layout: [0:3]=global_trans, [3:6]=global_orient, [6:136]=body_pose, [136:]=scale
    body_pose_good    = lbs_np[:, 6:136]    # (T, 130)
    global_trans_good  = lbs_np[:, 0:3]     # (T, 3)
    global_orient_good = lbs_np[:, 3:6]     # (T, 3)
    scale_params_good  = lbs_np[:, 136:204] # (T, 68)  bone-length scale params

    S, F = id_np.shape[1], expr_np.shape[1]
    SC = scale_params_good.shape[1]         # 68
    nan3   = np.full((T_full, 3),   np.nan, np.float32)
    nan130 = np.full((T_full, 130), np.nan, np.float32)
    nanS   = np.full((T_full, S),   np.nan, np.float32)
    nanF   = np.full((T_full, F),   np.nan, np.float32)
    nanSC  = np.full((T_full, SC),  np.nan, np.float32)

    body_pose_out     = nan130.copy(); body_pose_out[good_idx]     = body_pose_good
    global_trans_out  = nan3.copy();   global_trans_out[good_idx]  = global_trans_good
    global_orient_out = nan3.copy();   global_orient_out[good_idx] = global_orient_good
    scale_params_out  = nanSC.copy();  scale_params_out[good_idx]  = scale_params_good
    shape_params_out  = nanS.copy();   shape_params_out[good_idx]  = id_np
    expr_params_out   = nanF.copy();   expr_params_out[good_idx]   = expr_np

    # ------------------------------------------------------------------
    # 6) Save .npz  (same keys as original MHR inference output)
    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving → {out_npz}")
    save_dict = {
        "body_pose_params": body_pose_out,    # (T_full, 130)
        "global_trans":     global_trans_out, # (T_full, 3)
        "global_orient":    global_orient_out,# (T_full, 3)
        "scale_params":     scale_params_out, # (T_full, 68)  bone-length scales
        "shape_params":     shape_params_out, # (T_full, S)   identity blendshapes
        "expr_params":      expr_params_out,  # (T_full, F)
    }

    # ── Camera parameters ────────────────────────────────────────────────────
    # Camera params live at <subj_dir>/camera_parameters/<cam_id>/<action>.json,
    # alongside the SMPL-X JSON at      <subj_dir>/smplx/<action>.json.
    # We flatten each camera's params into the .npz as cam_<id>_* keys so that
    # downstream scripts (e.g. visualize_lab_dataset.py) can use them without a
    # separate lookup.
    _action_stem  = os.path.splitext(os.path.basename(args.smplx_json))[0]
    _subj_dir     = os.path.dirname(os.path.dirname(args.smplx_json))  # strip /smplx/
    _cam_root     = os.path.join(_subj_dir, "camera_parameters")
    _cams_saved   = []
    if os.path.isdir(_cam_root):
        for _cam_id in sorted(os.listdir(_cam_root)):
            _cam_json = os.path.join(_cam_root, _cam_id, f"{_action_stem}.json")
            if not os.path.isfile(_cam_json):
                continue
            with open(_cam_json) as _f:
                _cp = json.load(_f)
            save_dict[f"cam_{_cam_id}_ext_R"]     = np.array(_cp["extrinsics"]["R"],                    dtype=np.float64)  # (3,3)
            save_dict[f"cam_{_cam_id}_ext_T"]     = np.array(_cp["extrinsics"]["T"],                    dtype=np.float64)  # (1,3)
            save_dict[f"cam_{_cam_id}_intr_f"]    = np.array(_cp["intrinsics_w_distortion"]["f"],       dtype=np.float64)  # (1,2)
            save_dict[f"cam_{_cam_id}_intr_c"]    = np.array(_cp["intrinsics_w_distortion"]["c"],       dtype=np.float64)  # (1,2)
            save_dict[f"cam_{_cam_id}_intr_k"]    = np.array(_cp["intrinsics_w_distortion"]["k"],       dtype=np.float64)  # (1,3)
            save_dict[f"cam_{_cam_id}_intr_p"]    = np.array(_cp["intrinsics_w_distortion"]["p"],       dtype=np.float64)  # (1,2)
            save_dict[f"cam_{_cam_id}_intr_wo_f"] = np.array(_cp["intrinsics_wo_distortion"]["f"],      dtype=np.float64)  # (2,)
            save_dict[f"cam_{_cam_id}_intr_wo_c"] = np.array(_cp["intrinsics_wo_distortion"]["c"],      dtype=np.float64)  # (2,)
            _cams_saved.append(_cam_id)
        if _cams_saved:
            save_dict["camera_ids"] = np.array(_cams_saved)  # sorted cam id strings
    if not _cams_saved:
        print("   [WARN] No camera_parameters found — cam_* keys not saved to .npz")

    np.savez_compressed(out_npz, **save_dict)
    print(f"\n✅ Done. Saved {T_full} frames ({len(good_idx)} valid) to: {out_npz}")
    print(f"   body_pose_params : {body_pose_out.shape}")
    print(f"   scale_params     : {scale_params_out.shape}")
    print(f"   shape_params     : {shape_params_out.shape}")
    print(f"   expr_params      : {expr_params_out.shape}")
    if _cams_saved:
        print(f"   camera_params    : {_cams_saved}")

    # ── Optional: render comparison video  (video | MHR | SMPL-X) ───────────
    if out_video is not None:
        print("\n[VIDEO] Re-deriving MHR vertices from fitted params (sanity-check forward pass) ...")
        mhr_faces = np.asarray(mhr_model.character.mesh.faces, dtype=np.int32)

        # Scatter MHR params back to T_full (zeros for bad frames, then NaN-mask)
        lbs_full  = np.zeros((T_full, lbs_np.shape[1]), dtype=np.float32)
        id_full   = np.zeros((T_full, S), dtype=np.float32)
        expr_full = np.zeros((T_full, F), dtype=np.float32)
        lbs_full[good_idx]  = lbs_np
        id_full[good_idx]   = id_np
        expr_full[good_idx] = expr_np

        valid_frame_mask = np.zeros(T_full, dtype=bool)
        valid_frame_mask[good_idx] = True

        recon_list = []
        for start in range(0, T_full, args.batch_size):
            end = min(start + args.batch_size, T_full)
            with torch.no_grad():
                v, _ = mhr_model(
                    identity_coeffs  = torch.from_numpy(id_full[start:end]).to(device),
                    model_parameters = torch.from_numpy(lbs_full[start:end]).to(device),
                    face_expr_coeffs = torch.from_numpy(expr_full[start:end]).to(device),
                )
            recon_list.append(v.cpu().numpy())
        recon_verts = np.concatenate(recon_list, axis=0) / 100.0   # cm → m
        recon_verts[~valid_frame_mask] = np.nan                     # blank bad frames

        # Scatter SMPL-X verts to T_full (NaN for bad frames)
        smplx_verts_np = smplx_verts.cpu().numpy()                  # (T_good, V, 3) m
        V_smplx = smplx_verts_np.shape[1]
        smplx_verts_full = np.full((T_full, V_smplx, 3), np.nan, dtype=np.float32)
        smplx_verts_full[good_idx] = smplx_verts_np

        _render_comparison_video(
            mhr_verts=recon_verts,
            mhr_faces=mhr_faces,
            smplx_verts=smplx_verts_full,
            smplx_faces=smplx_faces,
            out_video=out_video,
            input_video=input_video,
            fps=args.video_fps,
            panel_h=args.video_panel_h,
        )


if __name__ == "__main__":
    main()
