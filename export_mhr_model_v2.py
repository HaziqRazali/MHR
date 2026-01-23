import torch
from mhr.mhr import MHR

class MHRWrapper(torch.nn.Module):
    def __init__(self, mhr):
        super().__init__()
        self.m = mhr
        self.skeleton = mhr.character_torch.skeleton
        self.parents = self.skeleton.joint_parents
        self.prerot = self.skeleton.joint_prerotations
        self.offsets = self.skeleton.joint_translation_offsets

    def forward(self, identity_coeffs, model_parameters, face_expr_coeffs):
        return self.m(identity_coeffs, model_parameters, face_expr_coeffs)

    def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: [...,4] in [x,y,z,w]
        ax, ay, az, aw = a.unbind(-1)
        bx, by, bz, bw = b.unbind(-1)
        rw = aw * bw - ax * bx - ay * by - az * bz
        rx = aw * bx + ax * bw + ay * bz - az * by
        ry = aw * by - ax * bz + ay * bw + az * bx
        rz = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack([rx, ry, rz, rw], dim=-1)

    def quat_inverse(q: torch.Tensor) -> torch.Tensor:
        # q: [...,4] -> inverse (conjugate / normsq)
        xyz = q[..., :3]
        w = q[..., 3:4]
        conj = torch.cat([-xyz, w], dim=-1)
        normsq = (q * q).sum(dim=-1, keepdim=True)
        return conj / (normsq + 1e-12)

    def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # rotate vector v by quaternion q: v' = v + 2*cross(q_xyz, cross(q_xyz, v) + q_w*v)
        q_xyz = q[..., :3]
        q_w = q[..., 3:4]
        t = 2.0 * torch.cross(q_xyz, v, dim=-1)
        return v + q_w * t + torch.cross(q_xyz, t, dim=-1)

    def quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
        # q: [...,4] in [x,y,z,w] -> returns [...,3] Euler angles (X, Y, Z) = (roll, pitch, yaw)
        x = q[..., 0]
        y = q[..., 1]
        z = q[..., 2]
        w = q[..., 3]

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        rx = torch.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        ry = torch.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        rz = torch.atan2(t3, t4)

        return torch.stack([rx, ry, rz], dim=-1)

    def skel_state_to_joint_parameters(self, skel_state):
        
        """
        Convert global skeleton state to flattened joint parameters.

        Args:
        skel_state: Tensor[..., J, 8] with layout [tx,ty,tz, qx,qy,qz,qw, scale]
        parents: LongTensor[J] with parent indices (root uses -1)
        prerot: Optional Tensor[J,4] quaternions [x,y,z,w] (or None for identity)
        offsets: Optional Tensor[J,3] joint_translation_offsets (or None for zeros)

        Returns:
        joint_parameters: Tensor[..., J*7] flattened per-joint [tx,ty,tz, rot_x,rot_y,rot_z, log2_scale]
        """

        # shapes
        *batch_shape, J, dim8 = list(skel_state.shape)
        assert dim8 == 8, "skel_state last dim must be 8"

        device = skel_state.device
        dtype = skel_state.dtype

        # Prepare tensors
        parent_idx = torch.clamp(self.parents, min=0).to(device=device)
        parent_skel = skel_state.index_select(-2, parent_idx)  # [..., J, 8]

        global_t = skel_state[..., :3]
        global_q = skel_state[..., 3:7]
        global_s = skel_state[..., 7:8]

        parent_t = parent_skel[..., :3]
        parent_q = parent_skel[..., 3:7]
        parent_s = parent_skel[..., 7:8]

        # inverse parent rotation
        inv_parent_q = self.quat_inverse(parent_q)

        # delta translation and local translation: rotate(inv_parent_q, global_t - parent_t) / parent_s
        delta_t = global_t - parent_t
        local_t = self.quat_rotate(inv_parent_q, delta_t) / (parent_s + 1e-12)
        # local rotation and scale
        local_q = self.quat_mul(inv_parent_q, global_q)
        local_s = global_s / (parent_s + 1e-12)

        # For joints with parent == -1, keep global as local (do not use computed values)
        parent_mask = (parents >= 0).to(device=device)  # [J]
        # expand mask to batch + joint dims
        expand_dims = [1] * (skel_state.ndim - 2) + [J, 1]  # e.g., for [..., J, 1]
        parent_mask_b = parent_mask.view(*([1] * (skel_state.ndim - 2)), J).to(device=device)
        parent_mask_b = parent_mask_b.expand(*((1,) * (skel_state.ndim - 2)), J).to(device=device).unsqueeze(-1)

        # Use torch.where to select computed local vs original global when parent < 0
        keep_local = parent_mask_b
        local_t = torch.where(keep_local, local_t, global_t)
        local_q = torch.where(keep_local, local_q, global_q)
        local_s = torch.where(keep_local, local_s, global_s)

        # subtract translation offsets if provided
        if self.offsets is None:
            translation_params = local_t
        else:
            # ensure offsets shape broadcast: [J,3] -> [1,...,J,3]
            translation_params = local_t - self.offsets.to(device=device, dtype=dtype)[None, ...] if skel_state.ndim > 2 else local_t - self.offsets.to(device=device, dtype=dtype)
        # remove prerotation if provided
        if self.prerot is None:
            adjusted_q = local_q
        else:
            prerot_b = self.prerot.to(device=device, dtype=dtype)
            # prerot_b: [J,4] -> broadcast
            adjusted_q = self.quat_mul(self.quat_inverse(prerot_b[None, ...] if skel_state.ndim > 2 else prerot_b), local_q)

        # convert quaternion -> euler XYZ
        rotation_params = self.quat_to_euler_xyz(adjusted_q)

        # scale param: log2(local_s)
        scale_params = torch.log2(local_s.clamp(min=1e-12))

        # concat per-joint params and flatten last two dims
        joint_params = torch.cat([translation_params, rotation_params, scale_params], dim=-1)  # [..., J, 7]
        return joint_params.flatten(-2, -1)

if __name__ == "__main__":
    # 1) load MHR (CPU). If you want the full model including pose correctives,
    # remove wants_pose_correctives=False
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

    # prepare example inputs (batch size 1) matching demo shapes
    num_id = mhr_model.get_num_identity_blendshapes()
    num_face = mhr_model.get_num_face_expression_blendshapes()
    full_size       = mhr_model.character.parameter_transform.size
    compact_size = int(full_size - (num_id + num_face))

    identity_coeffs = torch.zeros(1, num_id)
    model_parameters = torch.zeros(1, compact_size)
    face_expr_coeffs = torch.zeros(1, num_face)

    # 2) create wrapper with NOTHING extra
    wrapper = MHRWrapper(mhr_model)

    # 3) trace the wrapper's forward (delegates to Python MHR; trace records the ops)
    traced = torch.jit.trace(wrapper, (identity_coeffs, model_parameters, face_expr_coeffs))
    print("Traced wrapper forward.")

    # extract rig tensors from the Python MHR instance
    s = mhr_model.character_torch.skeleton
    parents = s.joint_parents
    prerot = s.joint_prerotations
    offsets = s.joint_translation_offsets

    # 4) Create a combined wrapper that embeds the traced forward and a scripted helper
    class CombinedWrapper(torch.nn.Module):
        def __init__(self, traced_module: torch.jit.ScriptModule, parents: torch.Tensor, prerot: torch.Tensor, offsets: torch.Tensor):
            super().__init__()
            # attach the already-traced module (ScriptModule)
            self.traced = traced_module
            # register rig buffers
            self.register_buffer("parents", parents.to(torch.long))
            self.register_buffer("prerot", prerot.to(torch.float32))
            self.register_buffer("offsets", offsets.to(torch.float32))

        def forward(self, identity_coeffs, model_parameters, face_expr_coeffs):
            return self.traced(identity_coeffs, model_parameters, face_expr_coeffs)

        @torch.jit.export
        def skel_state_to_joint_parameters(self, skel_state: torch.Tensor) -> torch.Tensor:
            # Expect skel_state shape [..., J, 8]
            J = skel_state.size(-2)
            dim8 = skel_state.size(-1)
            assert dim8 == 8
            device = skel_state.device
            dtype = skel_state.dtype

            parent_idx = torch.clamp(self.parents, min=0).to(device=device)
            parent_skel = skel_state.index_select(-2, parent_idx)

            global_t = skel_state[..., :3]
            global_q = skel_state[..., 3:7]
            global_s = skel_state[..., 7:8]

            parent_t = parent_skel[..., :3]
            parent_q = parent_skel[..., 3:7]
            parent_s = parent_skel[..., 7:8]

            # inverse parent rotation (conjugate / normsq)
            conj = torch.cat([-parent_q[..., :3], parent_q[..., 3:4]], dim=-1)
            normsq = (parent_q * parent_q).sum(dim=-1, keepdim=True)
            inv_parent_q = conj / (normsq + 1e-12)

            # rotate and compute local translation (broadcast-safe)
            q_xyz = inv_parent_q[..., :3]
            q_w = inv_parent_q[..., 3:4]
            t = 2.0 * torch.cross(q_xyz, (global_t - parent_t), dim=-1)
            local_t = (global_t - parent_t) + q_w * t + torch.cross(q_xyz, t, dim=-1)
            local_t = local_t / (parent_s + 1e-12)

            # local rotation and scale
            ax, ay, az, aw = inv_parent_q.unbind(-1)
            bx, by, bz, bw = global_q.unbind(-1)
            rw = aw * bw - ax * bx - ay * by - az * bz
            rx = aw * bx + ax * bw + ay * bz - az * by
            ry = aw * by - ax * bz + ay * bw + az * bx
            rz = aw * bz + ax * by - ay * bx + az * bw
            local_q = torch.stack([rx, ry, rz, rw], dim=-1)
            local_s = global_s / (parent_s + 1e-12)

            # build parent mask in a broadcast-friendly shape [J,1]
            parent_mask = (self.parents >= 0).to(device=device)
            parent_mask_b = parent_mask.view(J, 1).to(device=device)

            # select computed local vs original global when parent < 0 (broadcasting handles batch dims)
            local_t = torch.where(parent_mask_b, local_t, global_t)
            local_q = torch.where(parent_mask_b, local_q, global_q)
            local_s = torch.where(parent_mask_b, local_s, global_s)

            # subtract offsets if provided
            if self.offsets is not None:
                translation_params = local_t - self.offsets.to(device=device, dtype=dtype)[None, ...]
            else:
                translation_params = local_t

            # remove prerotation (prerot registered as buffer)
            p = self.prerot.to(device=device, dtype=dtype)
            p_conj = torch.cat([-p[..., :3], p[..., 3:4]], dim=-1)
            inv_pr = p_conj[None, ...]
            ax, ay, az, aw = inv_pr.unbind(-1)
            bx, by, bz, bw = local_q.unbind(-1)
            rw = aw * bw - ax * bx - ay * by - az * bz
            rx = aw * bx + ax * bw + ay * bz - az * by
            ry = aw * by - ax * bz + ay * bw + az * bx
            rz = aw * bz + ax * by - ay * bx + az * bw
            adjusted_q = torch.stack([rx, ry, rz, rw], dim=-1)

            # quaternion -> euler XYZ
            x = adjusted_q[..., 0]
            y = adjusted_q[..., 1]
            z = adjusted_q[..., 2]
            w = adjusted_q[..., 3]
            t0 = 2.0 * (w * x + y * z)
            t1 = 1.0 - 2.0 * (x * x + y * y)
            rx = torch.atan2(t0, t1)
            t2 = 2.0 * (w * y - z * x)
            t2 = torch.clamp(t2, -1.0, 1.0)
            ry = torch.asin(t2)
            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            rz = torch.atan2(t3, t4)
            rotation_params = torch.stack([rx, ry, rz], dim=-1)

            scale_params = torch.log2(local_s.clamp(min=1e-12))

            joint_params = torch.cat([translation_params, rotation_params, scale_params], dim=-1)
            return joint_params.flatten(-2, -1)

    # 5) script the combined wrapper so the saved .pt contains both forward and helper
    combined = CombinedWrapper(traced, parents, prerot, offsets)
    scripted_combined = torch.jit.script(combined)
    out_path = "mhr_model_v2.pt"
    torch.jit.save(scripted_combined, out_path)
    print("Saved", out_path)

    # verify by loading back
    loaded = torch.jit.load(out_path)
    verts_ref, skel_ref = wrapper(identity_coeffs, model_parameters, face_expr_coeffs)
    verts_loaded, skel_loaded = loaded(identity_coeffs, model_parameters, face_expr_coeffs)
    vdiff = (verts_ref - verts_loaded).abs().max().item()
    sdiff = (skel_ref - skel_loaded).abs().max().item()
    print(f"max vertex diff: {vdiff}")
    print(f"max skel diff: {sdiff}")

    jp = loaded.skel_state_to_joint_parameters(skel_loaded)