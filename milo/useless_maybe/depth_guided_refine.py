#!/usr/bin/env python3
"""Depth-guided refinement of Gaussian SDFs and mesh extraction.

This script optimizes a pretrained Gaussian Splat (PLY) using per-view depth maps.
Compared to `iterative_occupancy_refine.py`, Gaussian geometry is trainable and
the supervision comes directly from depth instead of RGB images.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from arguments import OptimizationParams, PipelineParams
from gaussian_renderer import integrate_radegs
from milo.useless_maybe.ply2mesh import (
    ManualScene,
    export_mesh_from_gaussians,
    initialize_mesh_regularization,
    load_cameras_from_json,
    build_render_functions,
)
from regularization.regularizer.mesh import compute_mesh_regularization
from scene.gaussian_model import GaussianModel, SparseGaussianAdam
from utils.general_utils import get_expon_lr_func
from torch.nn.utils import clip_grad_norm_


def ensure_learnable_occupancy(gaussians: GaussianModel) -> None:
    """Ensure occupancy buffers exist and the shift tensor is trainable."""
    if not gaussians.learn_occupancy or not hasattr(gaussians, "_occupancy_shift"):
        device = gaussians._xyz.device
        n_pts = gaussians._xyz.shape[0]
        base = torch.zeros((n_pts, 9), device=device)
        shift = torch.zeros_like(base)
        gaussians.learn_occupancy = True
        gaussians._base_occupancy = torch.nn.Parameter(base.requires_grad_(False), requires_grad=False)
        gaussians._occupancy_shift = torch.nn.Parameter(shift.requires_grad_(True))
    gaussians.set_occupancy_mode("occupancy_shift")
    gaussians._occupancy_shift.requires_grad_(True)


def extract_loss_scalars(metrics: Dict) -> Dict[str, float]:
    scalars: Dict[str, float] = {}
    for key, value in metrics.items():
        if not key.endswith("_loss"):
            continue
        scalar: Optional[float] = None
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                scalar = float(value.item())
        elif isinstance(value, (float, int)):
            scalar = float(value)
        if scalar is not None:
            scalars[key] = scalar
    return scalars


def export_iteration_state(
    iteration: int,
    gaussians: GaussianModel,
    mesh_state: Dict,
    output_dir: str,
    reference_camera=None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mesh_path = os.path.join(output_dir, f"mesh_iter_{iteration:05d}.ply")
    ply_path = os.path.join(output_dir, f"gaussians_iter_{iteration:05d}.ply")

    export_mesh_from_gaussians(
        gaussians=gaussians,
        mesh_state=mesh_state,
        output_path=mesh_path,
        reference_camera=reference_camera,
    )
    gaussians.save_ply(ply_path)


def natural_key(path: str) -> List[object]:
    """Split path into text/number tokens for natural sorting."""
    return [
        int(token) if token.isdigit() else token
        for token in re.split(r"(\d+)", path)
        if token
    ]


@dataclass
class DepthRecord:
    depth: torch.Tensor  # (1, H, W) on CPU
    valid_mask: torch.Tensor  # (1, H, W) on CPU, float mask in {0,1}


class DepthMapProvider:
    """Loads and serves depth maps corresponding to camera viewpoints."""

    def __init__(
        self,
        depth_dir: str,
        cameras: Sequence,
        depth_scale: float = 1.0,
        depth_offset: float = 0.0,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        if not os.path.isdir(depth_dir):
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
        self.depth_dir = depth_dir
        self.depth_scale = depth_scale
        self.depth_offset = depth_offset
        self.clip_min = clip_min
        self.clip_max = clip_max

        file_list = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
        if not file_list:
            raise ValueError(f"No depth npy files found in {depth_dir}")

        # Index depth files by (cam_idx, frame_idx) when possible.
        pattern = re.compile(r"depth_img_(\d+)_(\d+)\.npy$")
        indexed_files: Dict[Tuple[int, int], str] = {}
        for filename in file_list:
            match = pattern.match(filename)
            if match:
                cam_idx = int(match.group(1))
                frame_idx = int(match.group(2))
                indexed_files[(cam_idx, frame_idx)] = filename

        # Fallback: natural sorted list for sequential mapping.
        natural_sorted_files = sorted(file_list, key=natural_key)

        self.depth_height: Optional[int] = None
        self.depth_width: Optional[int] = None
        self.global_min: float = float("inf")
        self.global_max: float = float("-inf")
        self.global_valid_pixels: int = 0

        self.records: List[DepthRecord] = []
        for cam_idx, cam in enumerate(cameras):
            depth_path = self._resolve_path(
                cam.image_name if hasattr(cam, "image_name") else str(cam_idx),
                cam_idx,
                indexed_files,
                natural_sorted_files,
            )
            full_path = os.path.join(depth_dir, depth_path)
            depth_np = np.load(full_path)
            if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
                depth_np = depth_np[..., 0]
            if depth_np.ndim == 2:
                depth_np = depth_np[None, ...]  # (1, H, W)
            elif depth_np.ndim == 3 and depth_np.shape[0] == 1:
                pass  # already (1, H, W)
            else:
                raise ValueError(f"Unexpected depth shape {depth_np.shape} in {full_path}")

            depth_tensor = torch.from_numpy(depth_np.astype(np.float32))
            depth_tensor = depth_tensor * depth_scale + depth_offset

            if clip_min is not None or clip_max is not None:
                depth_tensor = depth_tensor.clamp(
                    min=clip_min if clip_min is not None else float("-inf"),
                    max=clip_max if clip_max is not None else float("inf"),
                )

            valid_mask = (depth_tensor > 0.0).float()
            # Track global statistics for diagnostics.
            if self.depth_height is None:
                self.depth_height, self.depth_width = depth_tensor.shape[-2:]
            valid_values = depth_tensor[valid_mask > 0.5]
            if valid_values.numel() > 0:
                self.global_min = min(self.global_min, float(valid_values.min().item()))
                self.global_max = max(self.global_max, float(valid_values.max().item()))
                self.global_valid_pixels += int(valid_values.numel())

            self.records.append(DepthRecord(depth=depth_tensor.contiguous(), valid_mask=valid_mask))

        if len(self.records) != len(cameras):
            raise RuntimeError("Depth map count does not match number of cameras.")
        if self.global_min == float("inf"):
            self.global_min = 0.0
            self.global_max = 0.0

    def _resolve_path(
        self,
        camera_name: str,
        camera_idx: int,
        indexed_files: Dict[Tuple[int, int], str],
        fallback_files: List[str],
    ) -> str:
        match = re.search(r"traj_(\d+)_cam(\d+)", camera_name)
        if match:
            frame_idx = int(match.group(1))
            cam_idx = int(match.group(2))
            candidate = indexed_files.get((cam_idx, frame_idx))
            if candidate:
                return candidate
        # Fallback to cam index with ordered list.
        if camera_idx >= len(fallback_files):
            raise IndexError(
                f"Camera index {camera_idx} exceeds depth file count {len(fallback_files)}."
            )
        return fallback_files[camera_idx]

    def get(self, index: int, device: torch.device) -> DepthRecord:
        record = self.records[index]
        depth = record.depth.to(device, non_blocking=True)
        valid = record.valid_mask.to(device, non_blocking=True)
        return DepthRecord(depth=depth, valid_mask=valid)


def compute_depth_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, float, float, int]:
    """Compute masked L1 loss and return (loss, mean_abs_error, valid_fraction, valid_pixels)."""
    if predicted.shape != target.shape:
        target = F.interpolate(
            target.unsqueeze(0),
            size=predicted.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        valid_mask = F.interpolate(
            valid_mask.unsqueeze(0),
            size=predicted.shape[-2:],
            mode="nearest",
        ).squeeze(0)

    valid = valid_mask > 0.5
    valid_pixels = valid.sum().item()
    if valid_pixels == 0:
        zero = torch.zeros((), device=predicted.device, dtype=predicted.dtype)
        return zero, 0.0, 0.0, 0

    diff = (predicted - target).abs() * valid_mask
    loss = diff.sum() / (valid_mask.sum() + epsilon)
    mae = diff.sum().item() / (valid_pixels + epsilon)
    valid_fraction = valid_pixels / valid_mask.numel()
    return loss, mae, valid_fraction, int(valid_pixels)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth-guided Gaussian refinement with mesh regularization.")
    parser.add_argument("--ply_path", type=str, required=True, help="Input Gaussian PLY.")
    parser.add_argument("--camera_poses", type=str, required=True, help="Camera pose JSON.")
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory containing per-view depth .npy files.")
    parser.add_argument("--mesh_config", type=str, default="medium", help="Mesh regularization config name.")
    parser.add_argument("--iterations", type=int, default=5000, help="Number of optimization steps.")
    parser.add_argument("--depth_loss_weight", type=float, default=1.0, help="(Deprecated) Depth loss multiplier; kept for backward compatibility.")
    parser.add_argument("--mesh_loss_weight", type=float, default=1.0, help="(Deprecated) Mesh loss multiplier; kept for backward compatibility.")
    parser.add_argument("--occupancy_lr_scale", type=float, default=1.0, help="Multiplier applied to occupancy LR.")
    parser.add_argument("--image_width", type=int, default=1280, help="Rendered image width.")
    parser.add_argument("--image_height", type=int, default=720, help="Rendered image height.")
    parser.add_argument("--fov_y", type=float, default=75.0, help="Vertical field-of-view in degrees.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="./depth_refine_output", help="Directory to store outputs.")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval.")
    parser.add_argument("--export_interval", type=int, default=1000, help="Mesh export interval (0 disables periodic export).")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Scale factor applied to loaded depth maps.")
    parser.add_argument("--depth_offset", type=float, default=0.0, help="Offset applied to loaded depth maps after scaling.")
    parser.add_argument("--depth_clip_min", type=float, default=None, help="Clip depth to minimum value (after scaling).")
    parser.add_argument("--depth_clip_max", type=float, default=None, help="Clip depth to maximum value (after scaling).")
    parser.add_argument("--freeze_colors", dest="freeze_colors", action="store_true", help="Freeze SH features during optimization.")
    parser.add_argument("--no-freeze_colors", dest="freeze_colors", action="store_false", help="Allow SH features to be optimized.")
    parser.set_defaults(freeze_colors=True)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0, help="Apply gradient clipping with given norm (0 disables).")
    parser.add_argument("--initial_lr_scale", type=float, default=1.0, help="Multiplier for position lr_init.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device.")
    parser.add_argument("--mesh_start_iter", type=int, default=1, help="Iteration at which mesh regularization starts.")
    parser.add_argument("--mesh_stop_iter", type=int, default=None, help="Optional iteration to stop mesh regularization.")
    parser.add_argument("--warn_until_iter", type=int, default=3000, help="Warmup iterations for surface sampling.")
    parser.add_argument("--imp_metric", type=str, default="outdoor", choices=["outdoor", "indoor"], help="Importance metric for surface sampling.")
    parser.add_argument("--depth_loss_epsilon", type=float, default=1e-6, help="Numerical epsilon for depth loss denominator.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device is required for depth-guided refinement.")

    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cameras = load_cameras_from_json(
        json_path=args.camera_poses,
        image_height=args.image_height,
        image_width=args.image_width,
        fov_y_deg=args.fov_y,
    )
    print(f"[INFO] Loaded {len(cameras)} cameras from {args.camera_poses}.")

    depth_provider = DepthMapProvider(
        depth_dir=args.depth_dir,
        cameras=cameras,
        depth_scale=args.depth_scale,
        depth_offset=args.depth_offset,
        clip_min=args.depth_clip_min,
        clip_max=args.depth_clip_max,
    )
    print(f"[INFO] Loaded {len(depth_provider.records)} depth maps from {args.depth_dir}.")
    if depth_provider.depth_height is not None:
        depth_h, depth_w = depth_provider.depth_height, depth_provider.depth_width
        if depth_h != args.image_height or depth_w != args.image_width:
            print(
                f"[WARNING] Depth resolution ({depth_w}x{depth_h}) differs from render resolution "
                f"({args.image_width}x{args.image_height}). Depth maps will be interpolated."
            )
    if depth_provider.global_valid_pixels == 0:
        print("[WARNING] No valid depth pixels found across dataset; depth supervision will be ineffective.")
    else:
        print(
            "[INFO] Depth value range after scaling: "
            f"{depth_provider.global_min:.4f} – {depth_provider.global_max:.4f} "
            f"({depth_provider.global_valid_pixels} valid pixels)."
        )

    scene = ManualScene(cameras)

    gaussians = GaussianModel(
        sh_degree=0,
        use_mip_filter=False,
        learn_occupancy=True,
        use_appearance_network=False,
    )
    gaussians.load_ply(args.ply_path)
    print(f"[INFO] Loaded {gaussians._xyz.shape[0]} Gaussians from {args.ply_path}.")

    ensure_learnable_occupancy(gaussians)
    gaussians.init_culling(len(cameras))
    if gaussians.spatial_lr_scale <= 0:
        gaussians.spatial_lr_scale = 1.0

    mesh_config = load_mesh_config(
        name=args.mesh_config,
        start_iter_override=args.mesh_start_iter,
        stop_iter_override=args.mesh_stop_iter,
        total_iterations=args.iterations,
    )
    occupancy_mode = mesh_config.get("occupancy_mode", "occupancy_shift")
    if occupancy_mode != "occupancy_shift":
        raise ValueError(
            f"Depth-guided refinement requires occupancy_mode 'occupancy_shift', got '{occupancy_mode}'. "
            "Please adjust the mesh configuration."
        )
    gaussians.set_occupancy_mode(occupancy_mode)
    print(
        "[INFO] Mesh config '{name}': start_iter={start}, stop_iter={stop}, n_max_points_in_delaunay={limit}".format(
            name=args.mesh_config,
            start=mesh_config.get("start_iter"),
            stop=mesh_config.get("stop_iter"),
            limit=mesh_config.get("n_max_points_in_delaunay"),
        )
    )

    opt_parser = argparse.ArgumentParser()
    opt_params = OptimizationParams(opt_parser)
    opt_params.iterations = args.iterations
    opt_params.position_lr_init *= args.initial_lr_scale
    opt_params.position_lr_final *= args.initial_lr_scale

    gaussians.training_setup(opt_params)

    if args.freeze_colors:
        gaussians._features_dc.requires_grad_(False)
        gaussians._features_rest.requires_grad_(False)

    lr_xyz_init = opt_params.position_lr_init * gaussians.spatial_lr_scale

    param_groups = [
        {"params": [gaussians._xyz], "lr": lr_xyz_init, "name": "xyz"},
        {"params": [gaussians._opacity], "lr": opt_params.opacity_lr, "name": "opacity"},
        {"params": [gaussians._scaling], "lr": opt_params.scaling_lr, "name": "scaling"},
        {"params": [gaussians._rotation], "lr": opt_params.rotation_lr, "name": "rotation"},
    ]
    if not args.freeze_colors:
        param_groups.append({"params": [gaussians._features_dc], "lr": opt_params.feature_lr, "name": "f_dc"})
        param_groups.append({"params": [gaussians._features_rest], "lr": opt_params.feature_lr / 20.0, "name": "f_rest"})
    if gaussians.learn_occupancy:
        param_groups.append({"params": [gaussians._occupancy_shift], "lr": opt_params.opacity_lr * args.occupancy_lr_scale, "name": "occupancy_shift"})

    gaussians.optimizer = SparseGaussianAdam(param_groups, lr=0.0, eps=1e-15)
    gaussians.xyz_scheduler_args = get_expon_lr_func(
        lr_init=lr_xyz_init,
        lr_final=opt_params.position_lr_final * gaussians.spatial_lr_scale,
        lr_delay_mult=opt_params.position_lr_delay_mult,
        max_steps=opt_params.position_lr_max_steps,
    )

    background = torch.zeros(3, dtype=torch.float32, device=device)
    pipe_parser = argparse.ArgumentParser()
    pipe = PipelineParams(pipe_parser)
    render_view, render_for_sdf = build_render_functions(gaussians, pipe, background)
    mesh_renderer, mesh_state = initialize_mesh_regularization(scene, mesh_config)
    mesh_state["reset_delaunay_samples"] = True
    mesh_state["reset_sdf_values"] = True

    runtime_args = argparse.Namespace(
        warn_until_iter=args.warn_until_iter,
        imp_metric=args.imp_metric,
        depth_reinit_iter=getattr(args, "depth_reinit_iter", args.warn_until_iter),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    loss_log_path = os.path.join(log_dir, "losses.jsonl")

    ema_depth_loss = None
    ema_mesh_loss = None
    pending_view_indices: List[int] = []
    printed_depth_diagnostics = False

    with open(loss_log_path, "w", encoding="utf-8") as loss_log_file:
        for iteration in range(1, args.iterations + 1):
            if not pending_view_indices:
                pending_view_indices = list(range(len(cameras)))
                random.shuffle(pending_view_indices)

            view_idx = pending_view_indices.pop()
            viewpoint = cameras[view_idx]

            depth_record = depth_provider.get(view_idx, device)
            render_pkg = render_view(viewpoint)

            pred_depth = render_pkg["median_depth"]
            depth_loss, depth_mae, valid_fraction, valid_pixels = compute_depth_loss(
                predicted=pred_depth,
                target=depth_record.depth,
                valid_mask=depth_record.valid_mask,
                epsilon=args.depth_loss_epsilon,
            )

            if valid_pixels == 0:
                skipped_record = {
                    "iteration": iteration,
                    "view_index": view_idx,
                    "skipped": True,
                    "skipped_reason": "invalid_depth",
                }
                loss_log_file.write(json.dumps(skipped_record) + "\n")
                loss_log_file.flush()
                if iteration % args.log_interval == 0 or iteration == 1:
                    print(f"[Iter {iteration:05d}] skipped view {view_idx} due to invalid depth.")
                continue

            total_loss = depth_loss

            mesh_pkg = compute_mesh_regularization(
                iteration=iteration,
                render_pkg=render_pkg,
                viewpoint_cam=viewpoint,
                viewpoint_idx=view_idx,
                gaussians=gaussians,
                scene=scene,
                pipe=pipe,
                background=background,
                kernel_size=0.0,
                config=mesh_config,
                mesh_renderer=mesh_renderer,
                mesh_state=mesh_state,
                render_func=render_for_sdf,
                weight_adjustment=100.0 / max(args.iterations, 1),
                args=runtime_args,
                integrate_func=integrate_radegs,
            )
            mesh_state = mesh_pkg["updated_state"]
            mesh_loss_tensor = mesh_pkg["mesh_loss"]
            mesh_loss = mesh_loss_tensor

            if not printed_depth_diagnostics:
                depth_valid = depth_record.depth[depth_record.valid_mask > 0.5]
                print(
                    "[DIAG] First valid depth batch: "
                    f"depth range {float(depth_valid.min().item()):.4f} – {float(depth_valid.max().item()):.4f}, "
                    f"predicted range {float(pred_depth.min().item()):.4f} – {float(pred_depth.max().item()):.4f}"
                )
                print(f"[DIAG] Gaussian spatial_lr_scale: {gaussians.spatial_lr_scale:.6f}")
                mesh_loss_unweighted = mesh_loss_tensor.item()
                mesh_loss_weighted_diag = mesh_loss.item()
                print(
                    f"[DIAG] Initial losses — depth_loss={depth_loss.item():.6e}, "
                    f"mesh_loss_raw={mesh_loss_unweighted:.6e}, "
                    f"mesh_loss_weighted={mesh_loss_weighted_diag:.6e}"
                )
                printed_depth_diagnostics = True

            total_loss = total_loss + mesh_loss

            gaussians.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.grad_clip_norm > 0.0:
                trainable_params: List[torch.Tensor] = []
                for group in gaussians.optimizer.param_groups:
                    for param in group.get("params", []):
                        if isinstance(param, torch.Tensor) and param.requires_grad:
                            trainable_params.append(param)
                if trainable_params:
                    clip_grad_norm_(trainable_params, args.grad_clip_norm)
            gaussians.update_learning_rate(iteration)
            visibility = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            gaussians.optimizer.step(visibility, radii.shape[0])

            total_loss_value = float(total_loss.item())
            depth_loss_value = float(depth_loss.item())
            mesh_loss_value = float(mesh_loss_tensor.item())
            weighted_mesh_loss_value = mesh_loss_value

            ema_depth_loss = depth_loss_value if ema_depth_loss is None else (0.9 * ema_depth_loss + 0.1 * depth_loss_value)
            ema_mesh_loss = weighted_mesh_loss_value if ema_mesh_loss is None else (0.9 * ema_mesh_loss + 0.1 * weighted_mesh_loss_value)

            iteration_record = {
                "iteration": iteration,
                "view_index": view_idx,
                "total_loss": total_loss_value,
                "depth_loss": depth_loss_value,
                "mesh_loss_raw": mesh_loss_value,
                "mesh_loss_weighted": weighted_mesh_loss_value,
                "ema_depth_loss": ema_depth_loss,
                "ema_mesh_loss": ema_mesh_loss,
                "depth_mae": depth_mae,
                "valid_fraction": valid_fraction,
                "valid_pixels": valid_pixels,
            }
            iteration_record.update(extract_loss_scalars(mesh_pkg))
            loss_log_file.write(json.dumps(iteration_record) + "\n")
            loss_log_file.flush()

            if args.export_interval > 0 and iteration % args.export_interval == 0:
                export_iteration_state(
                    iteration=iteration,
                    gaussians=gaussians,
                    mesh_state=mesh_state,
                    output_dir=args.output_dir,
                    reference_camera=None,
                )

            if iteration % args.log_interval == 0 or iteration == 1:
                print(
                    "[Iter {iter:05d}] loss={loss:.6f} depth={depth:.6f} mesh={mesh:.6f} "
                    "depth_mae={mae:.6f} valid={valid:.3f}".format(
                        iter=iteration,
                        loss=total_loss_value,
                        depth=depth_loss_value,
                        mesh=weighted_mesh_loss_value,
                        mae=depth_mae,
                        valid=valid_fraction,
                    )
                )

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    export_iteration_state(
        iteration=args.iterations,
        gaussians=gaussians,
        mesh_state=mesh_state,
        output_dir=final_dir,
        reference_camera=None,
    )
    print(f"[INFO] Depth-guided refinement completed. Results saved to {args.output_dir}.")


def load_mesh_config(
    name: str,
    start_iter_override: Optional[int] = None,
    stop_iter_override: Optional[int] = None,
    total_iterations: Optional[int] = None,
) -> Dict:
    from milo.useless_maybe.ply2mesh import load_mesh_config_file

    config = load_mesh_config_file(name)
    if start_iter_override is not None:
        config["start_iter"] = max(1, start_iter_override)
    else:
        config["start_iter"] = max(1, config.get("start_iter", 1))
    if stop_iter_override is not None:
        config["stop_iter"] = stop_iter_override
    elif total_iterations is not None:
        config["stop_iter"] = max(config.get("stop_iter", total_iterations), total_iterations)
    config["stop_iter"] = max(config.get("stop_iter", config["start_iter"]), config["start_iter"])
    return config


if __name__ == "__main__":
    main()
