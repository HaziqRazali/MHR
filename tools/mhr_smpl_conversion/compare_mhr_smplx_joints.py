# Compare MHR and SMPL-X skeletons in their default (T-) pose.
#
# For each SMPL-X joint referenced in smplx_to_t1.json, finds the closest
# MHR joint by Euclidean distance in the shared default pose space, then
# draws both skeletons side-by-side with labels.
#
# Usage (mhr_new env):
#   python compare_mhr_smplx_joints.py
#   python compare_mhr_smplx_joints.py --smplx_path /path/to/smplx

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch

# ── path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MHR_ROOT   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _MHR_ROOT not in sys.path:
    sys.path.insert(0, _MHR_ROOT)

from mhr.mhr import MHR

# ── IK joints from smplx_to_t1.json ─────────────────────────────────────────
# Maps SMPL-X joint index → name as it appears in the JSON
_SMPLX_IK_JOINTS = {
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

# ── Manual override mappings ─────────────────────────────────────────────────
# Override the geometric match for specific joints.
# Key   = SMPL-X joint name  (must be a value in _SMPLX_IK_JOINTS above)
# Value = MHR joint name     (must exist in the 127-joint skeleton)
# Example:  "spine3": "c_spine1"  forces spine3 → c_spine1 instead of c_spine3
_MANUAL_OVERRIDES: dict[str, str] = {
    "spine3": "c_spine3",
}

# ── SMPL-X skeleton edges (body only, 22 joints) ────────────────────────────
_SMPLX_EDGES = [
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

# ── MHR real (non-helper) joints we want to consider for matching ────────────
# Exclude twist_proc, _null helpers, and body_world (world-frame anchor, not a body joint)
_MHR_REAL = {
    1,                                           # root (pelvis)
    2, 3, 4, 5, 6, 7, 8,                         # l leg
    18, 19, 20, 21, 22, 23, 24,                  # r leg
    34, 35, 36, 37,                              # spine
    38, 39, 40, 41, 42,                          # r arm
    43, 44, 45, 46, 48, 49, 50, 52, 53, 54,     # r hand (skip nulls)
    56, 57, 58, 60, 61, 62, 63,
    74, 75, 76, 77, 78,                          # l arm
    79, 80, 81, 82, 84, 85, 86, 88, 89, 90,     # l hand (skip nulls)
    92, 93, 94, 96, 97, 98, 99,
    110, 113,                                    # neck, head
}

# Build MHR skeleton edges from parent table (filled at runtime)
def mhr_edges_from_parents(parents, keep):
    edges = []
    for j in keep:
        p = parents[j]
        while p != -1 and p not in keep:
            p = parents[p]
        if p != -1:
            edges.append((p, j))
    return edges


# ── helpers ──────────────────────────────────────────────────────────────────

def find_smplx_path():
    candidates = [
        "/home/haziq/datasets/mocap/data/models_smplx_v1_1/models/smplx",
        "/home/haziq/datasets/motion-x++/data/models_smplx_v1_1/models/smplx",
        "/media/haziq/Haziq/mocap/data/models_smplx_v1_1/models/smplx",
        os.path.expanduser("~/datasets/mocap/data/models_smplx_v1_1/models/smplx"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def build_lineset(pts, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color(color)
    return ls


def sphere_at(pos, radius=0.012, color=(1, 1, 0)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(pos)
    s.paint_uniform_color(list(color))
    return s


def _push_pos(pos, root_pos, push=0.25):
    """Return a position pushed outward from root_pos along root→joint ray."""
    d = np.array(pos, dtype=float) - np.array(root_pos, dtype=float)
    norm = np.linalg.norm(d)
    d = d / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    return np.array(pos, dtype=float) + d * push


def _decollide_y(label_pos_list, y_thresh=0.06, y_step=0.07):
    """
    Given a list of (label, pos, color), group entries whose pushed positions
    are within y_thresh of each other in Y, then spread them evenly in Y
    (alternating above/below the group centre).
    Returns a new list with adjusted positions.
    """
    items = [(lbl, np.array(p, dtype=float), col) for lbl, p, col in label_pos_list]
    used  = [False] * len(items)
    result = [None] * len(items)

    for i, (lbl_i, pos_i, col_i) in enumerate(items):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, len(items)):
            if not used[j] and abs(items[j][1][1] - pos_i[1]) < y_thresh:
                group.append(j)
        # spread group in Y around their mean Y
        mean_y = np.mean([items[k][1][1] for k in group])
        n = len(group)
        # sort group by X so symmetric joints go up/down consistently
        group.sort(key=lambda k: items[k][1][0])
        offsets = [(idx - (n - 1) / 2.0) * y_step for idx in range(n)]
        for rank, k in enumerate(group):
            lbl_k, pos_k, col_k = items[k]
            new_pos = pos_k.copy()
            new_pos[1] = mean_y + offsets[rank]
            result[k] = (lbl_k, new_pos, col_k)
            used[k] = True

    return result


def text_mesh_at(label, pos, scale=0.005, color=(0.2, 1.0, 0.2)):
    """Build a 3-D text mesh at the given position. Requires Open3D >= 0.16."""
    try:
        t = o3d.t.geometry.TriangleMesh.create_text(label, depth=0.0).to_legacy()
        t.scale(scale, center=t.get_center())
        t.translate(np.array(pos) - t.get_center())
        t.paint_uniform_color(list(color))
        return t
    except Exception:
        return None


def make_labels(entries, root_pos, scale=0.005, push=0.25):
    """
    entries: list of (label, joint_pos, color)
    Returns list of Open3D meshes with Y-decollided positions.
    """
    pushed = [(lbl, _push_pos(pos, root_pos, push), col)
              for lbl, pos, col in entries]
    decollided = _decollide_y(pushed)
    geoms = []
    for lbl, pos, col in decollided:
        t = text_mesh_at(lbl, pos, scale=scale, color=col)
        if t is not None:
            geoms.append(t)
    return geoms


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cpu")

    # ── 1. MHR default pose ──────────────────────────────────────────────────
    print("Loading MHR model …")
    mhr_model  = MHR.from_files(device=device, lod=1)
    skel       = mhr_model.character.skeleton
    mhr_names  = list(skel.joint_names)          # 127
    mhr_par    = list(skel.joint_parents)

    shape_p = torch.zeros(1, 45,  device=device)
    model_p = torch.zeros(1, 204, device=device)
    expr_p  = torch.zeros(1, 72,  device=device)

    with torch.no_grad():
        _, skel_state = mhr_model(shape_p, model_p, expr_p)

    # skel_state: (1, 127, 8),  [:3] = global XYZ in cm
    mhr_pos_cm = skel_state[0].cpu().numpy()[:, :3]   # (127, 3) cm
    mhr_pos    = mhr_pos_cm / 100.0                    # → metres

    # ── 2. SMPL-X default pose ───────────────────────────────────────────────
    import smplx

    smplx_path = args.smplx_path or find_smplx_path()
    if smplx_path is None:
        print("[ERROR] SMPL-X model path not found. Pass --smplx_path.")
        sys.exit(1)
    print(f"Loading SMPL-X from {smplx_path} …")

    smplx_model = smplx.SMPLX(
        model_path=smplx_path,
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)

    with torch.no_grad():
        out = smplx_model(
            betas            = torch.zeros(1, 10,  device=device),
            global_orient    = torch.zeros(1, 3,   device=device),
            body_pose        = torch.zeros(1, 63,  device=device),
            left_hand_pose   = torch.zeros(1, 45,  device=device),
            right_hand_pose  = torch.zeros(1, 45,  device=device),
            jaw_pose         = torch.zeros(1, 3,   device=device),
            leye_pose        = torch.zeros(1, 3,   device=device),
            reye_pose        = torch.zeros(1, 3,   device=device),
            expression       = torch.zeros(1, 10,  device=device),
        )

    smplx_pos = out.joints[0].cpu().numpy()   # (J, 3) metres  (J ≈ 127)

    # ── 3. Geometric matching ────────────────────────────────────────────────
    # Align by pelvis (SMPL-X joint 0) ↔ MHR root (joint 1).
    smplx_origin = smplx_pos[0].copy()
    mhr_origin   = mhr_pos[1].copy()

    smplx_centered = smplx_pos - smplx_origin
    mhr_centered   = mhr_pos   - mhr_origin

    # Scale: normalise by full body height  (root → head)
    # SMPL-X joint 15 = head,  MHR joint 113 = c_head
    smplx_body_h = np.linalg.norm(smplx_centered[15] - smplx_centered[0])
    mhr_head_idx = 113  # c_head
    mhr_body_h   = np.linalg.norm(mhr_centered[mhr_head_idx] - mhr_centered[1])
    scale = smplx_body_h / mhr_body_h if mhr_body_h > 1e-6 else 1.0
    mhr_scaled = mhr_centered * scale

    print(f"\nAlignment: SMPL-X body height={smplx_body_h:.3f} m, "
          f"MHR body height (scaled)={mhr_body_h*scale:.3f} m, scale={scale:.4f}")

    # For each SMPL-X IK joint → find closest MHR real joint
    mhr_real_list = sorted(_MHR_REAL)
    mhr_real_pos  = mhr_scaled[mhr_real_list]        # (N, 3)

    matches = {}   # smplx_idx → (mhr_idx, mhr_name, dist_m, is_override)
    for sx_idx, sx_name in _SMPLX_IK_JOINTS.items():
        sx_pt  = smplx_centered[sx_idx]
        dists  = np.linalg.norm(mhr_real_pos - sx_pt, axis=1)
        best_i = int(np.argmin(dists))
        mhr_idx  = mhr_real_list[best_i]
        mhr_name = mhr_names[mhr_idx]
        matches[sx_idx] = (mhr_idx, mhr_name, dists[best_i], False)

    # Apply manual overrides
    _sx_name_to_idx = {v: k for k, v in _SMPLX_IK_JOINTS.items()}
    for sx_name_key, mhr_name_override in _MANUAL_OVERRIDES.items():
        if sx_name_key not in _sx_name_to_idx:
            print(f"  [WARN] override key '{sx_name_key}' not in _SMPLX_IK_JOINTS — skipped")
            continue
        if mhr_name_override not in mhr_names:
            print(f"  [WARN] override target '{mhr_name_override}' not in MHR joint names — skipped")
            continue
        sx_idx   = _sx_name_to_idx[sx_name_key]
        mhr_idx  = mhr_names.index(mhr_name_override)
        sx_pt    = smplx_centered[sx_idx]
        dist     = float(np.linalg.norm(mhr_scaled[mhr_idx] - sx_pt))
        matches[sx_idx] = (mhr_idx, mhr_name_override, dist, True)
        print(f"  [OVERRIDE] {sx_name_key} → {mhr_name_override}  (dist={dist:.4f} m)")

    # ── 4. Print table ───────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print(f"{'SMPL-X idx':>10}  {'SMPL-X name':20}  {'MHR idx':>7}  "
          f"{'MHR name':30}  {'dist (m)':>8}")
    print("─"*75)
    for sx_idx, sx_name in _SMPLX_IK_JOINTS.items():
        mhr_idx, mhr_name, dist, is_override = matches[sx_idx]
        tag = " [OVERRIDE]" if is_override else ""
        print(f"{sx_idx:>10}  {sx_name:20}  {mhr_idx:>7}  {mhr_name:30}  {dist:>8.4f}{tag}")
    print("─"*75)

    # ── 5. Build Open3D geometry ─────────────────────────────────────────────
    GAP = 2   # metres between the two skeletons

    # ---- MHR skeleton (left, x = -GAP/2) -----------------------------------
    mhr_disp = mhr_scaled.copy()
    mhr_disp[:, 0] -= GAP / 2

    mhr_keep = sorted(_MHR_REAL)
    mhr_all_edges = mhr_edges_from_parents(mhr_par, set(mhr_keep))

    # remap edge indices to dense array for draw (use all 127 joints as pts)
    geoms = []
    geoms.append(build_lineset(mhr_disp, mhr_all_edges, color=[0.9, 0.2, 0.2]))

    matched_mhr_indices = {v[0] for v in matches.values()}

    for j in mhr_keep:
        is_matched = j in matched_mhr_indices
        r = 0.018 if is_matched else 0.009
        c = (1.0, 0.8, 0.0) if is_matched else (0.7, 0.3, 0.3)
        geoms.append(sphere_at(mhr_disp[j], radius=r, color=c))

    # Labels on matched MHR joints: show MHR name / smplx_name
    mhr_idx_to_smplx_name = {}
    for sx_idx, (mhr_idx, mhr_name, dist, is_override) in matches.items():
        sx_name = _SMPLX_IK_JOINTS[sx_idx]
        mhr_idx_to_smplx_name[mhr_idx] = (mhr_name, sx_name)

    mhr_root_pos = mhr_disp[1]
    mhr_entries  = [(f"{mhr_nm}/{sx_nm}", mhr_disp[mhr_idx], (1.0, 0.55, 0.0))
                    for mhr_idx, (mhr_nm, sx_nm) in mhr_idx_to_smplx_name.items()]
    geoms += make_labels(mhr_entries, root_pos=mhr_root_pos)

    # ---- SMPL-X skeleton (right, x = +GAP/2) --------------------------------
    sx_disp = smplx_centered.copy()
    sx_disp[:, 0] += GAP / 2

    n_body = 22
    geoms.append(build_lineset(sx_disp[:n_body], _SMPLX_EDGES, color=[0.3, 0.6, 1.0]))

    for sx_idx in range(n_body):
        is_ik = sx_idx in _SMPLX_IK_JOINTS
        r = 0.018 if is_ik else 0.009
        c = (0.2, 1.0, 0.4) if is_ik else (0.2, 0.4, 0.9)
        geoms.append(sphere_at(sx_disp[sx_idx], radius=r, color=c))

    sx_root_pos = sx_disp[0]
    sx_entries  = [(f"{sx_name}/{matches[sx_idx][1]}", sx_disp[sx_idx], (0.0, 0.9, 0.9))
                   for sx_idx, sx_name in _SMPLX_IK_JOINTS.items()
                   if sx_idx < len(sx_disp)]
    geoms += make_labels(sx_entries, root_pos=sx_root_pos)

    # ── 6. Axes + render ─────────────────────────────────────────────────────
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0]))

    print("\nOpening Open3D viewer  (press Q to quit, H for help)")
    print("  LEFT  (red)  : MHR default pose   — yellow spheres = matched joints")
    print("  RIGHT (blue) : SMPL-X default pose — green spheres  = IK joints from JSON")
    o3d.visualization.draw_geometries(
        geoms,
        window_name="MHR (left) vs SMPL-X (right) — default pose joint matching",
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise MHR & SMPL-X default-pose skeletons and find matching joints.",
    )
    parser.add_argument("--smplx_path", default=None,
                        help="Path to SMPL-X model folder (auto-detected if omitted)")
    main(parser.parse_args())
