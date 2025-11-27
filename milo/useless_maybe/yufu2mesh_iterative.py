#!/usr/bin/env python3
"""Iteratively optimize SDF pivots so that the extracted mesh adheres to the provided Gaussian point cloud."""

import argparse
import json
import math
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from arguments import PipelineParams
from functional import (
    sample_gaussians_on_surface,
    extract_gaussian_pivots,
    compute_initial_sdf_values,
    compute_delaunay_triangulation,
    extract_mesh,
)
from gaussian_renderer.radegs import render_radegs
from regularization.sdf.learnable import convert_sdf_to_occupancy
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel


def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to a rotation matrix."""
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def load_cameras(
    poses_json: str,
    height: int,
    width: int,
    fov_y_deg: float,
    device: str,
) -> List[Camera]:
    with open(poses_json, "r", encoding="utf-8") as f:
        poses = json.load(f)

    fov_y = math.radians(fov_y_deg)
    aspect = width / height
    fov_x = 2.0 * math.atan(aspect * math.tan(fov_y / 2.0))

    cameras: List[Camera] = []
    for idx, info in enumerate(poses):
        cam = Camera(
            colmap_id=str(idx),
            R=quaternion_to_rotation_matrix(info["quaternion"]),
            T=np.asarray(info["position"]),
            FoVx=fov_x,
            FoVy=fov_y,
            image=torch.empty(3, height, width),
            gt_alpha_mask=None,
            image_name=info.get("name", f"view_{idx:05d}"),
            uid=idx,
            data_device=device,
        )
        cameras.append(cam)
    return cameras


def build_render_function(
    gaussians: GaussianModel,
    pipe: PipelineParams,
    background: torch.Tensor,
):
    def render_func(view: Camera):
        render_pkg = render_radegs(
            viewpoint_camera=view,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            kernel_size=0.0,
            scaling_modifier=1.0,
            require_coord=False,
            require_depth=True,
        )
        return {"render": render_pkg["render"], "depth": render_pkg["median_depth"]}

    return render_func


def sample_tensor(tensor: torch.Tensor, max_samples: int) -> torch.Tensor:
    if max_samples <= 0 or tensor.shape[0] <= max_samples:
        return tensor
    idx = torch.randperm(tensor.shape[0], device=tensor.device)[:max_samples]
    return tensor[idx]


def export_mesh(mesh, path: str) -> None:
    verts = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


def main():
    parser = argparse.ArgumentParser(description="Iteratively refine SDF pivots using Chamfer supervision from the Gaussian cloud.")
    parser.add_argument("--ply_path", type=str, required=True, help="Perfect Gaussian PLY.")
    parser.add_argument("--camera_poses", type=str, required=True, help="Camera pose JSON.")
    parser.add_argument("--output_dir", type=str, default="./iter_occ_refine", help="Output directory.")
    parser.add_argument("--iterations", type=int, default=400, help="Number of SDF optimization steps.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for SDF pivots.")
    parser.add_argument("--reg_weight", type=float, default=5e-4, help="L2 regularization weight towards the initial SDF.")
    parser.add_argument("--mesh_sample_count", type=int, default=4096, help="Number of mesh vertices sampled per step.")
    parser.add_argument("--gaussian_sample_count", type=int, default=4096, help="Number of Gaussian centers sampled per step.")
    parser.add_argument("--surface_sample_limit", type=int, default=400000, help="Maximum Gaussians kept for Delaunay pivots.")
    parser.add_argument("--clamp_sdf", type=float, default=1.0, help="Clamp range for SDF values.")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval.")
    parser.add_argument("--export_interval", type=int, default=100, help="Mesh export interval (0 disables periodic export).")
    parser.add_argument("--image_height", type=int, default=720, help="Renderer image height.")
    parser.add_argument("--image_width", type=int, default=1280, help="Renderer image width.")
    parser.add_argument("--fov_y", type=float, default=75.0, help="Vertical FoV in degrees.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for occupancy refinement.")

    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    cameras = load_cameras(
        poses_json=args.camera_poses,
        height=args.image_height,
        width=args.image_width,
        fov_y_deg=args.fov_y,
        device=device,
    )

    gaussians = GaussianModel(
        sh_degree=0,
        use_mip_filter=False,
        learn_occupancy=True,
        use_appearance_network=False,
    )
    gaussians.load_ply(args.ply_path)

    pipe_parser = argparse.ArgumentParser()
    pipe = PipelineParams(pipe_parser)
    background = torch.tensor([0.0, 0.0, 0.0], device=device)
    render_func = build_render_function(gaussians, pipe, background)

    with torch.no_grad():
        means = gaussians.get_xyz.detach().clone()
        scales = gaussians.get_scaling.detach().clone()
        rotations = gaussians.get_rotation.detach().clone()

    with torch.no_grad():
        surface_gaussians_idx = sample_gaussians_on_surface(
            views=cameras,
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=gaussians.get_opacity,
            n_max_samples=args.surface_sample_limit,
            scene_type="outdoor",
        )

    if surface_gaussians_idx.numel() == 0:
        raise RuntimeError("Surface sampling returned zero Gaussians.")

    surface_means = means[surface_gaussians_idx].detach()

    initial_sdf = compute_initial_sdf_values(
        views=cameras,
        render_func=render_func,
        means=means,
        scales=scales,
        rotations=rotations,
        gaussian_idx=surface_gaussians_idx,
    ).detach()

    pivots, _ = extract_gaussian_pivots(
        means=means,
        scales=scales,
        rotations=rotations,
        gaussian_idx=surface_gaussians_idx,
    )

    delaunay_tets = compute_delaunay_triangulation(
        means=means,
        scales=scales,
        rotations=rotations,
        gaussian_idx=surface_gaussians_idx,
    )

    learned_sdf = torch.nn.Parameter(initial_sdf.clone())
    optimizer = torch.optim.Adam([learned_sdf], lr=args.lr)

    for iteration in range(1, args.iterations + 1):
        mesh = extract_mesh(
            delaunay_tets=delaunay_tets,
            pivots_sdf=learned_sdf,
            means=means,
            scales=scales,
            rotations=rotations,
            gaussian_idx=surface_gaussians_idx,
        )

        mesh_verts = mesh.verts
        if mesh_verts.numel() == 0:
            print(f"[Iter {iteration:05d}] Empty mesh, skipping update.")
            continue

        sampled_mesh_pts = sample_tensor(mesh_verts, args.mesh_sample_count)
        sampled_gaussian_pts = sample_tensor(surface_means, args.gaussian_sample_count)

        with torch.no_grad():
            nn_idx_forward = torch.cdist(
                sampled_mesh_pts.detach(),
                sampled_gaussian_pts.detach(),
                p=2,
            ).argmin(dim=1)
            nn_idx_backward = torch.cdist(
                sampled_gaussian_pts,
                sampled_mesh_pts.detach(),
                p=2,
            ).argmin(dim=1)

        nearest_gauss = sampled_gaussian_pts[nn_idx_forward]
        nearest_mesh = sampled_mesh_pts[nn_idx_backward]

        loss_forward = torch.mean(torch.sum((sampled_mesh_pts - nearest_gauss) ** 2, dim=1))
        loss_backward = torch.mean(torch.sum((sampled_gaussian_pts - nearest_mesh) ** 2, dim=1))
        chamfer_loss = loss_forward + loss_backward

        reg_loss = F.mse_loss(learned_sdf, initial_sdf)
        loss = chamfer_loss + args.reg_weight * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            learned_sdf.clamp_(-args.clamp_sdf, args.clamp_sdf)

        if iteration % args.log_interval == 0 or iteration == 1:
            print(
                f"[Iter {iteration:05d}] chamfer={chamfer_loss.item():.6f} "
                f"reg={reg_loss.item():.6f} total={loss.item():.6f} "
                f"|mesh|={sampled_mesh_pts.shape[0]} |gauss|={sampled_gaussian_pts.shape[0]}"
            )

        if args.export_interval > 0 and iteration % args.export_interval == 0:
            export_mesh(
                mesh=mesh,
                path=os.path.join(args.output_dir, f"mesh_iter_{iteration:05d}.ply"),
            )

    final_mesh = extract_mesh(
        delaunay_tets=delaunay_tets,
        pivots_sdf=learned_sdf,
        means=means,
        scales=scales,
        rotations=rotations,
        gaussian_idx=surface_gaussians_idx,
    )

    export_mesh(final_mesh, os.path.join(args.output_dir, "final_mesh.ply"))

    with torch.no_grad():
        final_occ = convert_sdf_to_occupancy(learned_sdf.detach()).view(-1, 9)
        base_occ = convert_sdf_to_occupancy(initial_sdf).view(-1, 9)
        gaussians.learn_occupancy = True
        total_gaussians = gaussians._xyz.shape[0]
        base_buffer = base_occ.new_zeros((total_gaussians, 9))
        shift_buffer = base_occ.new_zeros((total_gaussians, 9))
        surface_idx = surface_gaussians_idx.long()
        base_buffer.index_copy_(0, surface_idx, base_occ)
        shift_buffer.index_copy_(0, surface_idx, final_occ - base_occ)
        gaussians._base_occupancy = torch.nn.Parameter(base_buffer, requires_grad=False)
        gaussians._occupancy_shift = torch.nn.Parameter(shift_buffer, requires_grad=False)
        gaussians.save_ply(os.path.join(args.output_dir, "refined_gaussians.ply"))

    print(f"[INFO] Optimization complete. Results saved to {args.output_dir}.")


if __name__ == "__main__":
    main()
