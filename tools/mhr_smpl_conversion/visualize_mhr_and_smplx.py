# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Visualize an MHR mesh from a saved NPZ file by re-running the MHR forward pass.
# Does NOT use the pre-computed vertices stored in the NPZ.
#
# Only body_pose_params[:130] is used — global translation, global rotation,
# scale params, shape params, and expression params are all zeroed out.
# This isolates the pure body pose from the saved estimation.
#
# model_params (204,) layout passed to MHR:
#   [0:6]    zeros  — global trans (3) + global rot (3)
#   [6:136]  body_pose_params[:130]  — 130 body joint angles
#   [136:204] zeros  — scale params (68)
#   See: file:///home/haziq/MHR/demo.py
#
# Two NPZ formats are supported:
#
#   sam-3d-body format (single frame):
#     body_pose_params (133,)  — 130 body joint angles
#
#   fit3d / dataset format (multi-frame):
#     param_lbs_model_params  (T, 204) — full model params per frame
#     param_identity_coeffs   (T, 45)  — shape params per frame
#     param_face_expr_coeffs  (T, 72)  — expression params per frame
#     vertices                (T, V, 3) — pre-computed vertices (unused here)
#     Use --frame to select a frame (default 0).
#
# Usage:
#   python /home/haziq/MHR/tools/mhr_smpl_conversion/visualize_mhr_and_smplx.py --npz /home/haziq/sam-3d-body/example_data/results/img.npz --show_axes --convert_smplx
#   python /home/haziq/MHR/tools/mhr_smpl_conversion/visualize_mhr_and_smplx.py --npz /home/haziq/datasets/mocap/data/fit3d/train/s03/mhr/band_pull_apart.npz --frame 100 --show_axes --convert_smplx

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as ScipyR

from mhr.mhr import MHR
# file:///home/haziq/MHR/mhr/mhr.py

# The script lives in tools/mhr_smpl_conversion/ alongside conversion.py,
# so no extra sys.path manipulation is needed — just add the script's own dir.
_CONV_DIR = os.path.dirname(os.path.abspath(__file__))
if _CONV_DIR not in sys.path:
    sys.path.insert(0, _CONV_DIR)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_o3d_mesh(verts: np.ndarray, faces: np.ndarray, color=(0.75, 0.82, 0.95)):
    """Build an Open3D TriangleMesh from numpy arrays."""
    m = o3d.geometry.TriangleMesh()
    m.vertices  = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(faces.astype("int32"))
    m.compute_vertex_normals()
    m.paint_uniform_color(list(color))
    return m


def make_o3d_skeleton(joints: np.ndarray, edges: list, color=(1.0, 1.0, 0.0), y_offset: float = 1.5):
    """Build an Open3D LineSet skeleton from joint positions and edge pairs.
    y_offset: small vertical shift (metres) so the skeleton sits above the mesh.
    """
    pts = joints.copy()
    pts[:, 1] += y_offset
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color(list(color))
    return ls


def make_o3d_joint_spheres(
    joints: np.ndarray,
    color=(1.0, 1.0, 0.0),
    radius: float = 0.0125,
    y_offset: float = 1.5,
):
    """Build a merged TriangleMesh with one sphere per joint.
    y_offset must match the value used in make_o3d_skeleton (default 1.5 m).
    """
    merged = o3d.geometry.TriangleMesh()
    for pt in joints:
        pos = pt.copy()
        pos[1] += y_offset
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
        sphere.translate(pos)
        merged += sphere
    if len(merged.vertices):
        merged.compute_vertex_normals()
        merged.paint_uniform_color(list(color))
    return merged


# MHR joint indices to visualize (skip _proc, _twist_proc, _null helpers)
_MHR_DISPLAY = {
    1,                                       # root (pelvis) — body_world (0) is at origin, skip it
    2, 3, 4, 8,                              # L leg
    18, 19, 20, 24,                          # R leg
    34, 35, 36, 37,                          # spine
    38, 39, 40, 42,                          # R arm (clavicle→wrist)
    74, 75, 76, 78,                          # L arm
    110, 113,                                # neck, head
    # R hand
    43, 44, 45, 46,                          # r_pinky0-3
    48, 49, 50,                              # r_ring1-3
    52, 53, 54,                              # r_middle1-3
    56, 57, 58,                              # r_index1-3
    60, 61, 62, 63,                          # r_thumb0-3
    # L hand
    79, 80, 81, 82,                          # l_pinky0-3
    84, 85, 86,                              # l_ring1-3
    88, 89, 90,                              # l_middle1-3
    92, 93, 94,                              # l_index1-3
    96, 97, 98, 99,                          # l_thumb0-3
}

# SMPL-X body joint edges (indices into out.joints)
# 0=pelvis, 1=L-hip, 2=R-hip, 3=spine1, 4=L-knee, 5=R-knee,
# 6=spine2, 7=L-ankle, 8=R-ankle, 9=spine3, 10=L-foot, 11=R-foot,
# 12=neck, 13=L-collar, 14=R-collar, 15=head,
# 16=L-shoulder, 17=R-shoulder, 18=L-elbow, 19=R-elbow,
# 20=L-wrist, 21=R-wrist
_SMPLX_BODY_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5),
    (4, 7), (5, 8),
    (7, 10), (8, 11),
    (3, 6), (6, 9),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]
# SMPL-X hand joints (from joint_names.py):
#   25-39: L-hand  (index×3, middle×3, pinky×3, ring×3, thumb×3)
#   40-54: R-hand  (same finger order)
_SMPLX_HAND_EDGES = []
for _base, _wrist in [(25, 20), (40, 21)]:   # L-hand wrist=20, R-hand wrist=21
    for _f in range(5):                       # 5 fingers, 3 joints each
        _j0 = _base + _f * 3
        _SMPLX_HAND_EDGES.append((_wrist, _j0))
        _SMPLX_HAND_EDGES.append((_j0, _j0 + 1))
        _SMPLX_HAND_EDGES.append((_j0 + 1, _j0 + 2))
_SMPLX_EDGES = _SMPLX_BODY_EDGES + _SMPLX_HAND_EDGES

# Joint indices that get spheres: only those referenced in _SMPLX_EDGES.
# This excludes jaw (22), eyes (23-24), and any extra face landmarks (55+).
_SMPLX_SPHERE_JOINTS = sorted(set(i for e in _SMPLX_EDGES for i in e))

# SMPL-X joint indices whose names appear in smplx_to_t1.json
# (human_root_name + ik_match_table1/2 values)
_IK_LABEL_JOINTS = {
    0:  "pelvis",
    1:  "left_hip",
    2:  "right_hip",
    4:  "left_knee",
    5:  "right_knee",
    9:  "spine3",
    10: "left_foot",
    11: "right_foot",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
}


def make_smplx_joint_labels(
    joints_m: np.ndarray,
    x_offset: float = 0.0,
    y_offset: float = 1.5,
    text_scale: float = 0.0045,
    color=(0.2, 1.0, 0.2),
):
    """
    Build 3D text meshes for each SMPL-X joint that appears in smplx_to_t1.json.
    x_offset: applied to X so labels follow left/right/center layout shifts.
    y_offset: must match the y_offset used in make_o3d_skeleton (default 1.5 m).
    Requires Open3D >= 0.16 (o3d.t.geometry.TriangleMesh.create_text).
    """
    geoms = []
    try:
        for idx, name in _IK_LABEL_JOINTS.items():
            if idx >= len(joints_m):
                continue
            pos = joints_m[idx].copy()
            pos[1] += y_offset   # same shift as make_o3d_skeleton
            pos[0] += x_offset

            txt_mesh = o3d.t.geometry.TriangleMesh.create_text(name, depth=0.0)
            txt_leg  = txt_mesh.to_legacy()

            # create_text generates text ~1 unit tall; shrink to text_scale metres
            txt_leg.scale(text_scale, center=txt_leg.get_center())

            # Translate so the left edge of the text sits just right of the joint
            txt_leg.translate(pos - txt_leg.get_center() + np.array([0.03, 0.01, 0.0]))
            txt_leg.paint_uniform_color(list(color))
            geoms.append(txt_leg)
    except AttributeError:
        print("  [WARN] 3-D text labels require Open3D >= 0.16 — skipping joint labels.")
    return geoms


def mhr_joints_and_edges(skel_state_np: np.ndarray, parents: list):
    """
    skel_state_np: (127, 8) — global joint transforms, [:3] = position in cm
    parents:       list of 127 parent indices (-1 for root)
    Returns (joints_m, edges) where joints_m is (N, 3) in metres.

    For each displayed joint, we walk up the parent chain until we find another
    displayed joint, so intermediate helper joints (twist, null, proc) are
    skipped without leaving orphan nodes.
    """
    display = sorted(_MHR_DISPLAY)
    idx_map = {old: new for new, old in enumerate(display)}
    joints_m = skel_state_np[display, :3] / 100.0   # cm → metres

    edges = []
    for j in display:
        # walk up parent chain until we hit a displayed joint
        p = parents[j]
        while p != -1 and p not in idx_map:
            p = parents[p]
        if p != -1 and p in idx_map:
            edges.append((idx_map[p], idx_map[j]))
    return joints_m, edges


def find_smplx_path():
    candidates = [
        "/home/haziq/datasets/mocap/data/models_smplx_v1_1/models/smplx",
        "/home/haziq/datasets/motion-x++/data/models_smplx_v1_1/models/smplx",
        os.path.expanduser("/media/haziq/Haziq/mocap/data/models_smplx_v1_1/models/smplx"),
        os.path.expanduser("~/datasets/mocap/data/models_smplx_v1_1/models/smplx"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def expand_hand_pca_to_aa(smplx_model_pca, hand_pca, is_left=True):
    """Convert 6-dim PCA hand pose → full 45-dim axis-angle (15 joints × 3)."""
    if is_left:
        comps = smplx_model_pca.left_hand_components
        mean  = smplx_model_pca.left_hand_mean
    else:
        comps = smplx_model_pca.right_hand_components
        mean  = smplx_model_pca.right_hand_mean

    comps = comps.to(hand_pca.device, hand_pca.dtype)
    mean  = mean.to(hand_pca.device, hand_pca.dtype)

    if comps.shape[0] == hand_pca.shape[1]:
        aa45 = hand_pca @ comps
    else:
        aa45 = hand_pca @ comps.T

    return aa45 + mean.unsqueeze(0)  # (T, 45)


def convert_mhr_to_smplx(verts_m: np.ndarray, mhr_model, device, smplx_path: str):
    """
    Convert MHR vertices (metres) to SMPL-X vertices (metres).

    verts_m: (V, 3) numpy array in metres
    Returns smplx_verts_m (V_smplx, 3) numpy, smplx_faces (F, 3) numpy.
    """
    import smplx
    from conversion import Conversion

    # ── models ────────────────────────────────────────────────────────────
    smplx_model_pca = smplx.SMPLX(
        model_path=smplx_path,
        gender="neutral",
        use_pca=True,
        num_pca_comps=6,
        flat_hand_mean=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)

    # Script lives next to assets/, so ./assets/subsampled_vertex_indices.npy
    # resolves correctly without any chdir.
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model_pca,
        method="pytorch",
    )

    # converter expects (T, V, 3) in cm
    verts_cm = verts_m[None] * 100.0  # (1, V, 3)

    print("  Running MHR → SMPL-X conversion (PyTorch optimizer) ...")
    result = converter.convert_mhr2smpl(
        mhr_vertices=verts_cm,
        return_smpl_parameters=True,
    )
    params = result.result_parameters
    for k in params:
        v = params[k]
        params[k] = v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) \
                    else torch.tensor(v, device=device, dtype=torch.float32)

    T = next(iter(params.values())).shape[0]

    # expand PCA hands → full 45-dim AA
    params["left_hand_pose"]  = expand_hand_pca_to_aa(smplx_model_pca, params["left_hand_pose"],  is_left=True)
    params["right_hand_pose"] = expand_hand_pca_to_aa(smplx_model_pca, params["right_hand_pose"], is_left=False)

    # fill missing pose blocks
    params["jaw_pose"]  = torch.zeros((T, 3), device=device)
    params["leye_pose"] = torch.zeros((T, 3), device=device)
    params["reye_pose"] = torch.zeros((T, 3), device=device)

    if "betas" in params and params["betas"].shape[0] == T:
        params["betas"] = params["betas"][0:1]  # (1, 10)

    # ── forward with full hands ───────────────────────────────────────────
    smplx_model = smplx.SMPLX(
        model_path=smplx_path,
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)

    with torch.no_grad():
        out = smplx_model(**params)

    smplx_verts_m  = out.vertices[0].cpu().numpy()          # (V_smplx, 3) metres
    smplx_joints_m = out.joints[0].cpu().numpy()             # (J, 3) metres
    smplx_faces    = np.asarray(smplx_model.faces)           # (F, 3)

    # Build AMASS-compatible param dict for saving (all numpy, single frame)
    amass_params = {
        "root_orient":     params["global_orient"].detach().cpu().numpy(),   # (1, 3)
        "pose_body":       params["body_pose"].detach().cpu().numpy(),        # (1, 63)
        "trans":           params["transl"].detach().cpu().numpy(),           # (1, 3)
        "betas":           params["betas"].detach().cpu().numpy(),            # (1, 10)
        "left_hand_pose":  params["left_hand_pose"].detach().cpu().numpy(),   # (1, 45)
        "right_hand_pose": params["right_hand_pose"].detach().cpu().numpy(),  # (1, 45)
        "gender":          "neutral",
        "mocap_frame_rate": np.array(30.0),
    }
    return smplx_verts_m, smplx_faces, smplx_joints_m, amass_params


# MHR → SMPLX matched joint pairs used for orientation comparison
_MATCHED_PAIRS = [
    # (mhr_idx, mhr_name,    smplx_idx, smplx_name)
    (1,  "root",      0,  "pelvis"),
    (2,  "l_upleg",   1,  "left_hip"),
    (18, "r_upleg",   2,  "right_hip"),
    (3,  "l_lowleg",  4,  "left_knee"),
    (19, "r_lowleg",  5,  "right_knee"),
    (37, "c_spine3",  9,  "spine3"),
    (8,  "l_ball",    10, "left_foot"),
    (24, "r_ball",    11, "right_foot"),
    (75, "l_uparm",   16, "left_shoulder"),
    (39, "r_uparm",   17, "right_shoulder"),
    (76, "l_lowarm",  18, "left_elbow"),
    (40, "r_lowarm",  19, "right_elbow"),
]

# Fixed SMPL-X kinematic tree for the first 22 body joints
_SMPLX_PARENTS_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def _smplx_global_rots(root_orient_aa, body_pose_aa):
    """Chain SMPL-X local axis-angles into global rotations for the 22 body joints.
    root_orient_aa : (1, 3) or (3,)  numpy, axis-angle
    body_pose_aa   : (1, 63) or (63,) numpy, 21 × 3 axis-angle
    Returns list of 22 ScipyR objects (global orientations in world space).
    """
    root_aa = np.asarray(root_orient_aa).reshape(3)
    body_aa = np.asarray(body_pose_aa).reshape(63)
    n = len(_SMPLX_PARENTS_22)
    local = [ScipyR.from_rotvec(root_aa)]
    for i in range(1, n):
        local.append(ScipyR.from_rotvec(body_aa[(i - 1) * 3: i * 3]))
    global_rots = [None] * n
    global_rots[0] = local[0]
    for i in range(1, n):
        global_rots[i] = global_rots[_SMPLX_PARENTS_22[i]] * local[i]
    return global_rots


def print_global_orient_comparison(mhr_skel_np, smplx_amass_params):
    """Print a table comparing MHR vs SMPL-X global joint orientations for the
    same physical pose (after MHR→SMPL-X mesh conversion gave a perfect match).
    mhr_skel_np     : (127, 8) — skel_state[:,3:7] = xyzw quaternions in world space
    smplx_amass_params : dict with 'root_orient' (1,3) and 'pose_body' (1,63)
    """
    smplx_global = _smplx_global_rots(
        smplx_amass_params["root_orient"],
        smplx_amass_params["pose_body"],
    )

    W = 118
    print()
    print("=" * W)
    print("GLOBAL JOINT ORIENTATIONS — same physical pose  (MHR raw vs converted SMPL-X)")
    print("If the offset is constant across joints it is purely a local-frame definition difference.")
    print(f"{'Joint pair':<28} {'MHR global ZYX (deg)':<36} {'SMPLX global ZYX (deg)':<36} {'|diff| (deg)':>12}")
    print("-" * W)
    for mi, mn, si, sn in _MATCHED_PAIRS:
        mhr_rot   = ScipyR.from_quat(mhr_skel_np[mi, 3:7])   # skel stores xyzw
        smplx_rot = smplx_global[si]
        mhr_e     = mhr_rot.as_euler("zyx", degrees=True)
        smplx_e   = smplx_rot.as_euler("zyx", degrees=True)
        diff_deg  = (smplx_rot * mhr_rot.inv()).magnitude() * 180.0 / np.pi
        label     = f"{mn} ↔ {sn}"
        print(f"  {label:<26}  "
              f"MHR  =[{mhr_e[0]:+7.1f},{mhr_e[1]:+7.1f},{mhr_e[2]:+7.1f}]  "
              f"SMPLX=[{smplx_e[0]:+7.1f},{smplx_e[1]:+7.1f},{smplx_e[2]:+7.1f}]  "
              f"diff={diff_deg:7.1f}°")
    print("=" * W)
    print("NOTE: If diff is ~constant per joint → pure local-frame offset (fixable with rot_offset).")
    print("      If diff varies with pose → fundamentally different joint definitions (harder to fix).")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):

    # ── 1. Load NPZ ──────────────────────────────────────────────────────────
    print(f"Loading NPZ: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    print(f"  NPZ keys: {sorted(data.files)}")

    device = torch.device(args.device)

    if "body_pose_params" in data.files:
        # ── sam-3d-body single-frame format ──────────────────────────────────
        # body_pose_params (133,): 130 body joint angles + 3 jaw (always zero)
        # file:///home/haziq/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py
        body_pose = torch.tensor(data["body_pose_params"][:130], dtype=torch.float32)  # (130,)

        # Build model_params (204,) with only body joints, everything else zeroed:
        #   [0:6]    zeros — global trans (3) + global rot (3)
        #   [6:136]  body_pose_params[:130]
        #   [136:204] zeros — scale params (68)
        model_params = torch.zeros(1, 204, dtype=torch.float32)   # (1, 204)
        model_params[0, 6:136] = body_pose
        shape_params = torch.zeros(1, 45, dtype=torch.float32)    # (1, 45)
        expr_params  = torch.zeros(1, 72, dtype=torch.float32)    # (1, 72)

    elif "param_lbs_model_params" in data.files:
        # ── fit3d / dataset multi-frame format ───────────────────────────────
        # param_lbs_model_params (T, 204), param_identity_coeffs (T, 45),
        # param_face_expr_coeffs (T, 72)
        T = data["param_lbs_model_params"].shape[0]
        frame = args.frame
        print(f"  Sequence length: {T} frames")
        if frame < 0 or frame >= T:
            print(f"[ERROR] --frame {frame} is out of range [0, {T - 1}]")
            sys.exit(1)
        print(f"  Using frame {frame}")
        model_params = torch.tensor(data["param_lbs_model_params"][frame:frame+1], dtype=torch.float32)  # (1, 204)
        shape_params = torch.tensor(data["param_identity_coeffs"][frame:frame+1], dtype=torch.float32)  # (1, 45)
        expr_params  = torch.tensor(data["param_face_expr_coeffs"][frame:frame+1], dtype=torch.float32) # (1, 72)

    else:
        print("\n[ERROR] Unrecognised NPZ format.")
        print("  Expected either 'body_pose_params' (sam-3d-body) or")
        print("  'param_lbs_model_params' (fit3d/dataset multi-frame).")
        sys.exit(1)

    model_params = model_params.to(device)
    shape_params = shape_params.to(device)
    expr_params  = expr_params.to(device)

    # ── 2. Load MHR model ────────────────────────────────────────────────────
    # Uses assets at /home/haziq/MHR/assets/ by default (get_default_asset_folder())
    # file:///home/haziq/MHR/mhr/mhr.py
    print(f"Loading MHR model (device={device}, lod=1) ...")
    mhr_model = MHR.from_files(device=device, lod=1)
    mhr_faces  = mhr_model.character.mesh.faces   # (36874, 3) int32 numpy

    # ── 3. MHR forward pass ───────────────────────────────────────────────────
    # mhr_model(shape_params, model_params, expr_params)
    # → skinned_verts (B, 18439, 3) in **centimetres**, skel_state
    # file:///home/haziq/MHR/mhr/mhr.py
    #
    # NOTE: mhr_head.py also applies `verts[..., [1,2]] *= -1` (camera coord flip).
    #       Here we visualize in MHR's native coord space — rotate the viewer as needed.
    print("Running MHR forward pass ...")
    with torch.no_grad():
        verts, skel_state = mhr_model(shape_params, model_params, expr_params)

    verts    = verts[0]         # (18439, 3) cm
    verts    = verts / 100.0    # → metres
    verts_np = verts.cpu().numpy()
    print(f"  MHR verts: {verts_np.shape}  range [{verts_np.min():.3f}, {verts_np.max():.3f}] m")

    # Extract MHR joint positions from skel_state (global, cm)
    # skel_state: (B, 127, 8) — [:3] = global XYZ in cm
    mhr_skel_np = skel_state[0].cpu().numpy()                 # (127, 8)
    mhr_parents = list(mhr_model.character.skeleton.joint_parents)
    mhr_joints_m, mhr_edges = mhr_joints_and_edges(mhr_skel_np, mhr_parents)

    # ── 4. Build geometry ────────────────────────────────────────────────────
    mhr_mesh = make_o3d_mesh(verts_np, mhr_faces, color=(0.9, 0.2, 0.2))  # red
    geoms    = [mhr_mesh]
    if args.show_spheres:
        geoms.append(make_o3d_joint_spheres(mhr_joints_m, color=(1.0, 0.3, 0.3)))

    # ── 5. Optional: convert MHR → SMPL-X and overlay ────────────────────────
    if args.convert_smplx:
        smplx_path = args.smplx_path or find_smplx_path()
        if smplx_path is None:
            print("\n[ERROR] SMPL-X model path not found. Pass --smplx_path explicitly.")
            sys.exit(1)

        print(f"\nConverting MHR → SMPL-X  (smplx_path={smplx_path}) ...")
        smplx_verts, smplx_faces, smplx_joints_m, smplx_amass_params = convert_mhr_to_smplx(
            verts_np, mhr_model, device, smplx_path
        )
        print(f"  SMPL-X verts:  {smplx_verts.shape}   range [{smplx_verts.min():.3f}, {smplx_verts.max():.3f}] m")
        print(f"  SMPL-X joints: {smplx_joints_m.shape}")

        # ── Print global orientation comparison ───────────────────────────────
        print_global_orient_comparison(mhr_skel_np, smplx_amass_params)

        # Layout (left → right, gap = body_width + 0.3 m):
        #   LEFT   : MHR mesh + MHR skeleton
        #   CENTER : MHR + SMPL-X overlapping meshes + both skeletons + IK labels
        #   RIGHT  : SMPL-X mesh + SMPL-X skeleton + IK labels
        mesh_width = verts_np[:, 0].max() - verts_np[:, 0].min()
        gap        = mesh_width + 0.3

        # LEFT — MHR mesh + MHR skeleton
        mhr_verts_left = verts_np.copy()
        mhr_verts_left[:, 0] -= gap
        mhr_joints_left = mhr_joints_m.copy()
        mhr_joints_left[:, 0] -= gap
        geoms = [
            make_o3d_mesh(mhr_verts_left, mhr_faces, color=(0.9, 0.2, 0.2)),           # red mesh
            make_o3d_skeleton(mhr_joints_left, mhr_edges, color=(1.0, 0.3, 0.3)),       # red skel
        ]
        if args.show_spheres:
            geoms.append(make_o3d_joint_spheres(mhr_joints_left, color=(1.0, 0.3, 0.3)))

        # CENTER — MHR + SMPL-X overlapping meshes + both skeletons
        geoms += [
            make_o3d_mesh(verts_np,    mhr_faces,   color=(0.9, 0.2, 0.2)),             # red mesh
            make_o3d_mesh(smplx_verts, smplx_faces, color=(0.3, 0.6, 1.0)),             # blue mesh
            make_o3d_skeleton(mhr_joints_m,   mhr_edges,    color=(1.0, 0.3, 0.3)),     # red skel
            make_o3d_skeleton(smplx_joints_m, _SMPLX_EDGES, color=(0.3, 0.6, 1.0)),    # blue skel
        ]
        # spheres intentionally omitted for the center overlap panel

        # RIGHT — SMPL-X mesh + SMPL-X skeleton + IK labels
        smplx_verts_right = smplx_verts.copy()
        smplx_verts_right[:, 0] += gap
        smplx_joints_right = smplx_joints_m.copy()
        smplx_joints_right[:, 0] += gap
        geoms += [
            make_o3d_mesh(smplx_verts_right, smplx_faces, color=(0.3, 0.6, 1.0)),       # blue mesh
            make_o3d_skeleton(smplx_joints_right, _SMPLX_EDGES, color=(0.3, 0.6, 1.0)), # blue skel
        ]
        if args.show_spheres:
            geoms.append(make_o3d_joint_spheres(smplx_joints_right[_SMPLX_SPHERE_JOINTS], color=(0.3, 0.6, 1.0)))
        # IK joint name labels on right SMPL-X skeleton
        if not args.no_labels:
            geoms += make_smplx_joint_labels(smplx_joints_m, x_offset=gap)

        print(f"  Layout: MHR (left, -{gap:.2f} m) | MHR+SMPL-X+skeletons+labels (center) | SMPL-X+labels (right, +{gap:.2f} m)")

        if args.save_smplx is not None:
            save_dir = os.path.dirname(os.path.abspath(args.save_smplx))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            np.savez(
                args.save_smplx,
                # AMASS-compatible parameters (loadable by load_smplx_file)
                root_orient=smplx_amass_params["root_orient"],
                pose_body=smplx_amass_params["pose_body"],
                trans=smplx_amass_params["trans"],
                betas=smplx_amass_params["betas"],
                left_hand_pose=smplx_amass_params["left_hand_pose"],
                right_hand_pose=smplx_amass_params["right_hand_pose"],
                gender=smplx_amass_params["gender"],
                mocap_frame_rate=smplx_amass_params["mocap_frame_rate"],
                # Mesh output for direct use
                vertices=smplx_verts,
                faces=smplx_faces,
                joints=smplx_joints_m,
            )
            print(f"  Saved SMPL-X results to: {args.save_smplx}")

    if args.show_axes:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]))

    # ── 6. Show ───────────────────────────────────────────────────────────────
    label = "MHR alone | MHR+SMPL-X overlap | SMPL-X alone" if args.convert_smplx else "MHR Mesh"
    print(f"\nOpening Open3D viewer  [{label}]  (press Q to quit, H for help) ...")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=label,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize MHR mesh from a saved NPZ file using the MHR forward pass.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_mhr_and_smplx.py --npz /home/haziq/sam-3d-body/example_data/results/img.npz --convert_smplx --device cuda
  python visualize_mhr_and_smplx.py --npz /home/haziq/sam-3d-body/example_data/results/img.npz --convert_smplx --smplx_path /path/to/smplx
  python visualize_mhr_and_smplx.py --npz /home/haziq/sam-3d-body/example_data/results/img.npz --convert_smplx --device cuda --save_smplx /home/haziq/sam-3d-body/example_data/results/img_smplx.npz
        """,
    )
    parser.add_argument("--npz",           required=True,       help="Path to .npz file: sam-3d-body single-frame or fit3d multi-frame format")
    parser.add_argument("--device",        default="cpu",       help="Torch device: cpu or cuda (default: cpu)")
    parser.add_argument("--frame",         type=int, default=0, help="Frame index for multi-frame NPZ files (default: 0)")
    parser.add_argument("--show_axes",     action="store_true", help="Show coordinate frame axes")
    parser.add_argument("--show_spheres",  action="store_true", help="Show skeleton joints as spheres (off by default)")
    parser.add_argument("--no_labels",     action="store_true", help="Hide SMPL-X joint name labels (shown by default)")
    parser.add_argument("--convert_smplx", action="store_true", help="Convert MHR mesh to SMPL-X and show side by side")
    parser.add_argument("--smplx_path",    default=None,        help="Path to SMPL-X model folder (auto-detected if omitted)")
    parser.add_argument("--save_smplx",    default=None,        help="Path to save SMPL-X results as .npz (vertices, faces, joints). Requires --convert_smplx.")
    args = parser.parse_args()
    main(args)
