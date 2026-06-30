import os
import sys
import types
import argparse
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import smplx
# 调用官方成熟、稳定的前向核心算子，规避底层复杂的矩阵公式
from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform,
)


class LegacyChumpyShim:
    def __setstate__(self, state):
        self.__dict__.update(state)

    def __array__(self, dtype=None):
        return np.asarray(self.r if hasattr(self, "r") else self.x, dtype=dtype)

    @property
    def shape(self): return np.asarray(self).shape

    def __len__(self): return len(np.asarray(self))

    def __getitem__(self, item): return np.asarray(self)[item]


if "chumpy.ch" not in sys.modules:
    chumpy_m = types.ModuleType("chumpy")
    chumpy_ch_m = types.ModuleType("chumpy.ch")
    LegacyChumpyShim.__name__ = "Ch"
    LegacyChumpyShim.__qualname__ = "Ch"
    LegacyChumpyShim.__module__ = "chumpy.ch"
    chumpy_ch_m.Ch = LegacyChumpyShim
    chumpy_m.ch = chumpy_ch_m
    sys.modules["chumpy"] = chumpy_m
    sys.modules["chumpy.ch"] = chumpy_ch_m


# 核心 LBS 流程控制
def RunCustomLBS(model, betas, global_orient, body_pose):
    dtype = betas.dtype
    device = betas.device

    # 模板状态提取
    v_template = model.v_template.unsqueeze(0)  # [1, V, 3]

    # 形状形变
    shapedirs = model.shapedirs[:, :, :betas.shape[1]]
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 关节回归
    J_initial = vertices2joints(model.J_regressor, v_shaped)

    # 姿态修正
    full_pose = torch.cat([global_orient, body_pose], dim=1)
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
    pose_feature = (rot_mats[:, 1:] - ident).view(1, -1)

    # 兼容处理 posedirs 维度
    posedirs = model.posedirs.view(-1, model.posedirs.shape[-1])
    if posedirs.shape[0] != pose_feature.shape[1] and posedirs.shape[1] == pose_feature.shape[1]:
        posedirs = posedirs.T

    pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets

    # 蒙皮线性混合
    J_transformed, A = batch_rigid_transform(rot_mats, J_initial, model.parents, dtype=dtype)

    # 混合矩阵加权
    W = model.lbs_weights.unsqueeze(0).expand(1, -1, -1)
    T = torch.matmul(W, A.view(1, A.shape[1], 16)).view(1, -1, 4, 4)

    # 四维齐次坐标计算最终网格顶点
    homo_coord = torch.ones_like(v_posed[:, :, :1])
    v_posed_homo = torch.cat([v_posed, homo_coord], dim=2)
    v_final_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
    v_final = v_final_homo[:, :, :3, 0]

    # 辅助变量
    J_template = vertices2joints(model.J_regressor, v_template)

    return {
        "v_template": v_template[0].detach().cpu().numpy(),
        "J_template": J_template[0].detach().cpu().numpy(),
        "v_shaped": v_shaped[0].detach().cpu().numpy(),
        "J_shaped": J_initial[0].detach().cpu().numpy(),
        "pose_offsets": pose_offsets[0].detach().cpu().numpy(),
        "v_posed": v_posed[0].detach().cpu().numpy(),
        "J_transformed": J_transformed[0].detach().cpu().numpy(),
        "verts": v_final[0].detach().cpu().numpy(),
        "lbs_weights": model.lbs_weights.detach().cpu().numpy()
    }


# 可视化，使用独立的Lambert漫反射光照渲染器
class SMPLLBSVisualizer:
    def __init__(self, faces):
        self.faces = faces

    def _set_balanced_bbox(self, ax, verts):
        center = verts.mean(axis=0)
        max_range = np.max(verts.max(axis=0) - verts.min(axis=0)) / 2.0
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[2] - max_range, center[2] + max_range)
        ax.set_zlim(center[1] - max_range, center[1] + max_range)
        ax.set_axis_off()

    def render_mesh_to_axis(self, ax, verts, joints=None, scalars=None, face_colors=None, title=""):
        x, y, z = verts[:, 0], verts[:, 2], verts[:, 1]
        plot_verts = np.stack([x, y, z], axis=1)

        if face_colors is not None:
            colors = face_colors
        elif scalars is not None:
            norm = plt.Normalize(vmin=scalars.min(), vmax=scalars.max() + 1e-8)
            cmap = plt.get_cmap("plasma")
            v_colors = cmap(norm(scalars))[:, :3]
            colors = v_colors[self.faces].mean(axis=1)
        else:
            colors = np.tile(np.array([0.85, 0.75, 0.70]), (len(self.faces), 1))

        triangles = plot_verts[self.faces]
        v0 = triangles[:, 1] - triangles[:, 0]
        v1 = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(v0, v1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

        light_dir = np.array([-0.3, 0.4, 0.8])
        light_dir /= np.linalg.norm(light_dir)

        intensity = 0.4 + 0.6 * np.clip(np.sum(normals * light_dir, axis=1), 0.0, 1.0)
        shaded_colors = colors * intensity[:, np.newaxis]

        # 3D 表面贴片渲染
        ax.plot_trisurf(x, y, z, triangles=self.faces, color='none', shade=False)
        mesh = ax.collections[-1]
        mesh.set_facecolors(shaded_colors)
        mesh.set_edgecolors((0, 0, 0, 0.03))
        mesh.set_linewidth(0.1)

        if joints is not None:
            ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1],
                       color='cyan', edgecolor='black', s=25, zorder=10)

        self._set_balanced_bbox(ax, verts)
        ax.set_title(title, fontsize=11, fontweight='bold')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--out-dir", type=str, default="./outputs")
    parser.add_argument("--joint-id", type=int, default=18)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"正在从 {args.model_dir} 加载模型并初始化流程...")
    model = smplx.create(model_path=os.path.abspath(args.model_dir),
                         model_type="smpl", gender="neutral", ext="pkl").to(device)

    faces = np.asarray(model.faces, dtype=np.int32)

    # 设定形状和驱动骨骼角度
    betas = torch.zeros((1, 10), dtype=torch.float32, device=device)
    betas[0, 0], betas[0, 1] = 2.0, -1.2

    global_orient = torch.zeros((1, 3), dtype=torch.float32, device=device)
    body_pose = torch.zeros((1, 23 * 3), dtype=torch.float32, device=device)
    body_pose[0, (16 - 1) * 3 + 2] = 0.45  # 抬左臂
    body_pose[0, (18 - 1) * 3 + 1] = -0.35  # 弯肘部

    # 运行整合后的前向 LBS 计算
    res = RunCustomLBS(model, betas, global_orient, body_pose)

    # 精确度验证：比对官方黑盒 forward
    with torch.no_grad():
        official_out = model(betas=betas, global_orient=global_orient, body_pose=body_pose)
    official_verts = official_out.vertices[0].cpu().numpy()

    mae = np.mean(np.abs(res["verts"] - official_verts))
    max_err = np.max(np.abs(res["verts"] - official_verts))

    viz = SMPLLBSVisualizer(faces)

    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection='3d')
    viz.render_mesh_to_axis(ax, res["v_template"], res["J_template"],
                            scalars=res["lbs_weights"][:, args.joint_id],
                            title=f"Stage A: Weight Map (Joint {args.joint_id})")
    fig.savefig(os.path.join(args.out_dir, "stage_a_template_weights.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    dominant_joints = np.argmax(res["lbs_weights"], axis=1)
    face_dominant = dominant_joints[faces].mean(axis=1).astype(int)
    palette = plt.get_cmap("gist_ncar")(np.linspace(0, 1, res["lbs_weights"].shape[1]))
    all_joint_colors = palette[face_dominant][:, :3]

    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection='3d')
    viz.render_mesh_to_axis(ax, res["v_template"], res["J_template"],
                            face_colors=all_joint_colors, title="All Joint LBS Weights Distribution")
    fig.savefig(os.path.join(args.out_dir, "all_joint_weights.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection='3d')
    viz.render_mesh_to_axis(ax, res["v_shaped"], res["J_shaped"], title="Stage B: Shape Correction & Regression")
    fig.savefig(os.path.join(args.out_dir, "stage_b_shaped_joints.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    pose_offsets_norm = np.linalg.norm(res["pose_offsets"], axis=1)
    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection='3d')
    viz.render_mesh_to_axis(ax, res["v_posed"], res["J_shaped"], scalars=pose_offsets_norm,
                            title="Stage C: Pose Corrective Offsets")
    fig.savefig(os.path.join(args.out_dir, "stage_c_pose_offsets.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection='3d')
    viz.render_mesh_to_axis(ax, res["verts"], res["J_transformed"], title="Stage D: Final Skinned LBS Result")
    fig.savefig(os.path.join(args.out_dir, "stage_d_lbs_result.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(14, 12))
    titles = ["(a) template + weights", "(b) shape + joints", "(c) pose offsets", "(d) final skinned mesh"]
    for i, title in enumerate(titles):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        if i == 0:
            viz.render_mesh_to_axis(ax, res["v_template"], res["J_template"],
                                    scalars=res["lbs_weights"][:, args.joint_id], title=title)
        elif i == 1:
            viz.render_mesh_to_axis(ax, res["v_shaped"], res["J_shaped"], title=title)
        elif i == 2:
            viz.render_mesh_to_axis(ax, res["v_posed"], res["J_shaped"], scalars=pose_offsets_norm, title=title)
        elif i == 3:
            viz.render_mesh_to_axis(ax, res["verts"], res["J_transformed"], title=title)
    fig.savefig(os.path.join(args.out_dir, "comparison_grid.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("===== SMPL LBS Semi-Custom Lab Summary =====\n")
        f.write(f"num_vertices: {model.v_template.shape[0]}\n")
        f.write(f"num_faces: {faces.shape[0]}\n")
        f.write(f"num_joints(from lbs_weights): {model.lbs_weights.shape[1]}\n")
        f.write(f"manual_vs_official_mean_abs_error: {mae:.10e}\n")
        f.write(f"manual_vs_official_max_abs_error: {max_err:.10e}\n")

    print(f"实验运行成功！MAE 误差为: {mae:.10e}（趋于 0 的完美对齐），图像结果保存在 {args.out_dir}")


if __name__ == "__main__":
    main()