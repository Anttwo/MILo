#!/usr/bin/env python3
"""
Depth-supervised training loop for 3D Gaussian Splatting.

This script mirrors the original MILo image-supervised training pipeline, but
replaces the photometric loss with a depth reconstruction objective fed by
per-view depth maps. It supports mesh-in-the-loop regularization, gaussian
densification/simplification, and periodic exports for inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
import trimesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(ROOT_DIR)

from arguments import OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render_simp  # noqa: E402
from gaussian_renderer.radegs import render_radegs as render_radegs  # noqa: E402
from gaussian_renderer.radegs import integrate_radegs as integrate  # noqa: E402
from regularization.regularizer.mesh import initialize_mesh_regularization  # noqa: E402
from regularization.regularizer.mesh import compute_mesh_regularization  # noqa: E402
from regularization.regularizer.mesh import reset_mesh_state_at_next_iteration  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.geometry_utils import flatten_voronoi_features  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from functional import extract_mesh, compute_delaunay_triangulation  # noqa: E402
from functional.mesh import frustum_cull_mesh  # noqa: E402
from regularization.sdf.learnable import convert_occupancy_to_sdf  # noqa: E402


def quaternion_to_rotation_matrix(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,)")
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    rotation = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return rotation


def load_cameras_from_json(
    json_path: str,
    image_height: int,
    image_width: int,
    fov_y_deg: float,
    data_device: str,
) -> List[Camera]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Camera JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not entries:
        raise ValueError(f"No camera entries in {json_path}")

    fov_y = math.radians(fov_y_deg)
    aspect = image_width / image_height
    fov_x = 2.0 * math.atan(aspect * math.tan(fov_y * 0.5))

    cameras: List[Camera] = []
    for idx, entry in enumerate(entries):
        if "quaternion" in entry:
            rotation = quaternion_to_rotation_matrix(entry["quaternion"])
        elif "rotation" in entry:
            rotation = np.asarray(entry["rotation"], dtype=np.float32)
            if rotation.shape != (3, 3):
                raise ValueError(f"Camera entry {idx} rotation must be 3x3")
        else:
            raise KeyError(f"Camera entry {idx} missing rotation or quaternion.")

        if "tvec" in entry:
            translation = np.asarray(entry["tvec"], dtype=np.float32)
        elif "translation" in entry:
            translation = np.asarray(entry["translation"], dtype=np.float32)
        elif "position" in entry:
            camera_center = np.asarray(entry["position"], dtype=np.float32)
            if camera_center.shape != (3,):
                raise ValueError(f"Camera entry {idx} position must be length-3.")
            translation = -rotation.T @ camera_center
        else:
            raise KeyError(f"Camera entry {idx} missing translation/position.")

        if translation.shape != (3,):
            raise ValueError(f"Camera entry {idx} translation must be length-3.")

        image_name = (
            entry.get("name")
            or entry.get("img_name")
            or entry.get("image_name")
            or f"view_{idx:04d}"
        )

        camera = Camera(
            colmap_id=str(idx),
            R=rotation,
            T=translation,
            FoVx=fov_x,
            FoVy=fov_y,
            image=torch.zeros(3, image_height, image_width),
            gt_alpha_mask=None,
            image_name=image_name,
            uid=idx,
            data_device=data_device,
        )
        cameras.append(camera)
    return cameras


def _clone_camera_for_scale(camera: Camera, scale: float) -> Camera:
    if math.isclose(scale, 1.0):
        return camera

    new_height = max(1, int(round(camera.image_height / scale)))
    new_width = max(1, int(round(camera.image_width / scale)))
    blank_image = torch.zeros(3, new_height, new_width, dtype=torch.float32)

    # Camera expects rotation/translation as numpy arrays; reuse existing values.
    return Camera(
        colmap_id=camera.colmap_id,
        R=camera.R,
        T=camera.T,
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=blank_image,
        gt_alpha_mask=None,
        image_name=camera.image_name,
        uid=camera.uid,
        data_device=str(camera.data_device),
    )


def _build_scaled_cameras(
    cameras: Sequence[Camera],
    scales: Sequence[float] = (1.0, 2.0),
) -> Dict[float, List[Camera]]:
    scaled: Dict[float, List[Camera]] = {}
    for scale in scales:
        if math.isclose(scale, 1.0):
            scaled[float(scale)] = list(cameras)
        else:
            scaled[float(scale)] = [_clone_camera_for_scale(cam, scale) for cam in cameras]
    return scaled


class ManualScene:
    """Minimal adapter exposing camera access expected by mesh regularizer."""

    def __init__(self, cameras_by_scale: Dict[float, Sequence[Camera]]):
        if 1.0 not in cameras_by_scale:
            raise ValueError("At least scale 1.0 cameras must be provided.")
        self._train_cameras: Dict[float, List[Camera]] = {
            float(scale): list(cam_list) for scale, cam_list in cameras_by_scale.items()
        }

    def getTrainCameras(self, scale: float = 1.0):
        scale_key = float(scale)
        if scale_key not in self._train_cameras:
            scale_key = 1.0
        return list(self._train_cameras[scale_key])

    def getTrainCameras_warn_up(
        self,
        iteration: int,
        warn_until_iter: int,
        scale: float = 1.0,
        scale2: float = 2.0,
    ):
        preferred = scale2 if iteration <= warn_until_iter and float(scale2) in self._train_cameras else scale
        fallback_scale = float(preferred) if float(preferred) in self._train_cameras else 1.0
        return list(self._train_cameras[fallback_scale])


def build_render_functions(
    gaussians: GaussianModel,
    pipe: PipelineParams,
    background: torch.Tensor,
):
    def _render(
        view: Camera,
        pc_obj: GaussianModel,
        pipe_obj: PipelineParams,
        bg_color: torch.Tensor,
        *,
        kernel_size: float = 0.0,
        require_coord: bool = False,
        require_depth: bool = True,
    ):
        pkg = render_radegs(
            viewpoint_camera=view,
            pc=pc_obj,
            pipe=pipe_obj,
            bg_color=bg_color,
            kernel_size=kernel_size,
            scaling_modifier=1.0,
            require_coord=require_coord,
            require_depth=require_depth,
        )
        if "area_max" not in pkg:
            pkg["area_max"] = torch.zeros_like(pkg["radii"])
        return pkg

    def render_view(view: Camera):
        return _render(view, gaussians, pipe, background)

    def render_for_sdf(
        view: Camera,
        gaussians_override: Optional[GaussianModel] = None,
        pipeline_override: Optional[PipelineParams] = None,
        background_override: Optional[torch.Tensor] = None,
        kernel_size: float = 0.0,
        require_depth: bool = True,
        require_coord: bool = False,
    ):
        pc_obj = gaussians if gaussians_override is None else gaussians_override
        pipe_obj = pipe if pipeline_override is None else pipeline_override
        bg_color = background if background_override is None else background_override
        pkg = _render(
            view,
            pc_obj,
            pipe_obj,
            bg_color,
            kernel_size=kernel_size,
            require_coord=require_coord,
            require_depth=require_depth,
        )
        return {
            "render": pkg["render"].detach(),
            "median_depth": pkg["median_depth"].detach(),
        }

    return render_view, render_for_sdf


def export_mesh_from_gaussians(
    gaussians: GaussianModel,
    mesh_state: Dict,
    output_path: str,
    reference_camera: Optional[Camera] = None,
) -> None:
    delaunay_tets = mesh_state.get("delaunay_tets")
    gaussian_idx = mesh_state.get("delaunay_xyz_idx")
    if delaunay_tets is None:
        delaunay_tets = compute_delaunay_triangulation(
            means=gaussians.get_xyz,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            gaussian_idx=gaussian_idx,
        )

    occupancy = (
        gaussians.get_occupancy
        if gaussian_idx is None
        else gaussians.get_occupancy[gaussian_idx]
    )
    pivots_sdf = convert_occupancy_to_sdf(flatten_voronoi_features(occupancy))

    mesh = extract_mesh(
        delaunay_tets=delaunay_tets,
        pivots_sdf=pivots_sdf,
        means=gaussians.get_xyz,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        gaussian_idx=gaussian_idx,
    )

    mesh_to_export = mesh
    if reference_camera is not None:
        mesh_to_export = frustum_cull_mesh(mesh, reference_camera)

    verts = mesh_to_export.verts.detach().cpu().numpy()
    faces = mesh_to_export.faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(output_path)


def load_mesh_config_file(name: str) -> Dict:
    config_path = os.path.join(BASE_DIR, "configs", "mesh", f"{name}.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Mesh config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_learnable_occupancy(gaussians: GaussianModel) -> None:
    """Ensure occupancy buffers exist and shifts are trainable."""
    if not gaussians.learn_occupancy or not hasattr(gaussians, "_occupancy_shift"):
        device = gaussians._xyz.device
        n_pts = gaussians._xyz.shape[0]
        base = torch.zeros((n_pts, 9), device=device)
        shift = torch.zeros_like(base)
        gaussians.learn_occupancy = True
        gaussians._base_occupancy = torch.nn.Parameter(
            base.requires_grad_(False), requires_grad=False
        )
        gaussians._occupancy_shift = torch.nn.Parameter(shift.requires_grad_(True))
    gaussians.set_occupancy_mode("occupancy_shift")
    gaussians._occupancy_shift.requires_grad_(True)


@dataclass
class DepthRecord:
    depth: torch.Tensor  # (1, H, W)
    valid_mask: torch.Tensor  # (1, H, W)


class DepthMapProvider:
    """Loads depth maps and matches them to cameras via naming convention."""

    def __init__(
        self,
        depth_dir: Path,
        cameras: Sequence,
        depth_scale: float = 1.0,
        depth_offset: float = 0.0,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

        file_list = sorted([f.name for f in depth_dir.iterdir() if f.suffix == ".npy"])
        if not file_list:
            raise ValueError(f"No depth .npy files found in {depth_dir}")

        pattern = re.compile(r"depth_img_(\d+)_(\d+)\.npy$")
        indexed: Dict[Tuple[int, int], str] = {}
        for filename in file_list:
            match = pattern.match(filename)
            if match:
                cam_idx = int(match.group(1))
                frame_idx = int(match.group(2))
                indexed[(cam_idx, frame_idx)] = filename

        fallback_files = sorted(file_list, key=self._natural_key)

        self.depth_scale = depth_scale
        self.depth_offset = depth_offset
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.depth_height: Optional[int] = None
        self.depth_width: Optional[int] = None
        self.global_min: float = float("inf")
        self.global_max: float = float("-inf")
        self.global_valid_pixels: int = 0
        self.records: List[DepthRecord] = []

        for cam_index, camera in enumerate(cameras):
            depth_path = self._resolve_path(
                camera_name=getattr(camera, "image_name", str(cam_index)),
                camera_idx=cam_index,
                indexed_files=indexed,
                fallback_files=fallback_files,
            )
            full_path = depth_dir / depth_path
            depth_np = np.load(full_path)
            if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
                depth_np = depth_np[..., 0]
            if depth_np.ndim == 2:
                depth_np = depth_np[None, ...]
            elif depth_np.ndim == 3 and depth_np.shape[0] == 1:
                pass
            else:
                raise ValueError(f"Unexpected depth shape {depth_np.shape} in {full_path}")

            depth = torch.from_numpy(depth_np.astype(np.float32))
            depth = depth * depth_scale + depth_offset
            if clip_min is not None or clip_max is not None:
                depth = depth.clamp(
                    min=clip_min if clip_min is not None else float("-inf"),
                    max=clip_max if clip_max is not None else float("inf"),
                )

            mask = (depth > 0.0).float()
            if self.depth_height is None:
                self.depth_height, self.depth_width = depth.shape[-2:]

            valid_values = depth[mask > 0.5]
            if valid_values.numel() > 0:
                self.global_min = min(self.global_min, float(valid_values.min()))
                self.global_max = max(self.global_max, float(valid_values.max()))
                self.global_valid_pixels += int(valid_values.numel())

            self.records.append(DepthRecord(depth=depth.contiguous(), valid_mask=mask))

        if self.global_min == float("inf"):
            self.global_min = 0.0
            self.global_max = 0.0

    @staticmethod
    def _natural_key(path: str) -> List[object]:
        tokens = re.split(r"(\d+)", Path(path).stem)
        return [int(tok) if tok.isdigit() else tok for tok in tokens if tok]

    @staticmethod
    def _resolve_path(
        camera_name: str,
        camera_idx: int,
        indexed_files: Dict[Tuple[int, int], str],
        fallback_files: Sequence[str],
    ) -> str:
        match = re.search(r"traj_(\d+)_cam(\d+)", camera_name)
        if match:
            frame_idx = int(match.group(1))
            cam_idx = int(match.group(2))
            candidate = indexed_files.get((cam_idx, frame_idx))
            if candidate:
                return candidate
        if camera_idx >= len(fallback_files):
            raise IndexError(
                f"Camera index {camera_idx} exceeds number of depth files ({len(fallback_files)})."
            )
        return fallback_files[camera_idx]

    def __len__(self) -> int:
        return len(self.records)

    def get(self, index: int, device: torch.device) -> DepthRecord:
        record = self.records[index]
        return DepthRecord(
            depth=record.depth.to(device, non_blocking=True),
            valid_mask=record.valid_mask.to(device, non_blocking=True),
        )


def compute_depth_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, float, float, int]:
    if predicted.shape != target.shape:
        target = F.interpolate(
            target.unsqueeze(0),
            size=predicted.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0),
            size=predicted.shape[-2:],
            mode="nearest",
        ).squeeze(0)

    valid = mask > 0.5
    valid_pixels = int(valid.sum().item())
    if valid_pixels == 0:
        zero = torch.zeros((), device=predicted.device, dtype=predicted.dtype)
        return zero, 0.0, 0.0, 0

    diff = (predicted - target).abs() * mask
    loss = diff.sum() / (mask.sum() + epsilon)
    mae = diff.sum().item() / (valid_pixels + epsilon)
    valid_fraction = valid_pixels / mask.numel()
    return loss, mae, valid_fraction, valid_pixels


class DepthTrainer:
    """Orchestrates depth-supervised optimization of a Gaussian model."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self._prepare_seeds(args.seed)

        base_cameras = load_cameras_from_json(
            json_path=args.camera_poses,
            image_height=args.image_height,
            image_width=args.image_width,
            fov_y_deg=args.fov_y,
            data_device=args.data_device,
        )
        print(f"[INFO] Loaded {len(base_cameras)} cameras.")
        self.cameras_by_scale = _build_scaled_cameras(base_cameras, scales=(1.0, 2.0))
        self.scene = ManualScene(self.cameras_by_scale)
        self.cameras = self.cameras_by_scale[1.0]

        self.fixed_view_idx = args.fixed_view_idx
        if self.fixed_view_idx is not None:
            if not (0 <= self.fixed_view_idx < len(self.cameras)):
                raise ValueError(
                    f"fixed_view_idx {self.fixed_view_idx} out of bounds for {len(self.cameras)} cameras."
                )

        depth_dir = Path(args.depth_dir)
        self.depth_provider = DepthMapProvider(
            depth_dir=depth_dir,
            cameras=self.cameras,
            depth_scale=args.depth_scale,
            depth_offset=args.depth_offset,
            clip_min=args.depth_clip_min,
            clip_max=args.depth_clip_max,
        )
        if self.depth_provider.global_valid_pixels == 0:
            raise RuntimeError("No valid depth pixels found across the dataset.")
        print(
            "[INFO] Depth statistics after scaling: "
            f"{self.depth_provider.global_min:.4f} – {self.depth_provider.global_max:.4f} "
            f"({self.depth_provider.global_valid_pixels} valid pixels)"
        )

        self.scene.cameras_extent = self._estimate_extent(args.ply_path)

        self.gaussians = GaussianModel(
            sh_degree=args.sh_degree,
            use_mip_filter=not args.disable_mip_filter,
            learn_occupancy=True,
            use_appearance_network=False,
        )
        self.gaussians.load_ply(args.ply_path)
        ensure_learnable_occupancy(self.gaussians)
        self.gaussians.init_culling(len(self.cameras))
        if self.gaussians.spatial_lr_scale <= 0:
            self.gaussians.spatial_lr_scale = 1.0

        opt_parser = argparse.ArgumentParser(add_help=False)
        opt_params = OptimizationParams(opt_parser)
        opt_params.iterations = args.iterations
        opt_params.position_lr_init *= args.initial_lr_scale
        opt_params.position_lr_final *= args.initial_lr_scale
        self.gaussians.training_setup(opt_params)
        if args.freeze_colors:
            if hasattr(self.gaussians, "_features_dc"):
                self.gaussians._features_dc.requires_grad_(False)
            if hasattr(self.gaussians, "_features_rest"):
                self.gaussians._features_rest.requires_grad_(False)

        self.background = torch.zeros(3, dtype=torch.float32, device=self.device)
        pipe_parser = argparse.ArgumentParser(add_help=False)
        self.pipe = PipelineParams(pipe_parser)
        self.pipe.compute_cov3D_python = args.compute_cov3d_python
        self.pipe.convert_SHs_python = args.convert_shs_python
        self.pipe.debug = args.debug

        self.render_view, self.render_for_mesh = build_render_functions(
            self.gaussians, self.pipe, self.background
        )
        self.mesh_enabled = args.mesh_regularization
        if self.mesh_enabled:
            mesh_config = self._load_mesh_config(
                args.mesh_config, args.mesh_start_iter, args.mesh_stop_iter, args.iterations
            )
            occupancy_mode = mesh_config.get("occupancy_mode", "occupancy_shift")
            if occupancy_mode != "occupancy_shift":
                raise ValueError(
                    f"Mesh config '{args.mesh_config}' must use occupancy_mode 'occupancy_shift', got '{occupancy_mode}'."
                )
            self.gaussians.set_occupancy_mode(occupancy_mode)
            self.mesh_renderer, self.mesh_state = initialize_mesh_regularization(
                self.scene,
                mesh_config,
            )
            self.mesh_state["reset_delaunay_samples"] = True
            self.mesh_state["reset_sdf_values"] = True
            self.mesh_config = mesh_config
            self.runtime_args = argparse.Namespace(
                warn_until_iter=args.warn_until_iter,
                imp_metric=args.imp_metric,
                depth_reinit_iter=args.depth_reinit_iter,
            )
            self._warmup_mesh_visibility()
        else:
            self.mesh_renderer = None
            self.mesh_state = {}
            self.mesh_config = {}
            self.runtime_args = None

        self.optimizer = self.gaussians.optimizer
        self.opt_params = opt_params

        self.output_dir = Path(args.output_dir)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.loss_log_path = self.output_dir / "logs" / "losses.jsonl"
        self.pending_indices: List[int] = []
        self.ema_depth: Optional[float] = None
        self.ema_mesh: Optional[float] = None
        self.printed_depth_diag = False
        self.log_depth_stats = bool(args.log_depth_stats or self.fixed_view_idx is not None)

    @staticmethod
    def _prepare_seeds(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _estimate_extent(self, ply_path: str) -> float:
        import trimesh

        mesh = trimesh.load(ply_path, process=False)
        if hasattr(mesh, "vertices"):
            vertices = np.asarray(mesh.vertices)
            center = vertices.mean(axis=0)
            radius = np.linalg.norm(vertices - center, axis=1).max()
            return float(radius)
        raise ValueError("Could not estimate scene extent from PLY.")

    def _load_mesh_config(
        self,
        name: str,
        start_iter_override: Optional[int],
        stop_iter_override: Optional[int],
        total_iterations: int,
    ) -> Dict:
        config = load_mesh_config_file(name)
        if start_iter_override is not None:
            config["start_iter"] = max(1, start_iter_override)
        if stop_iter_override is not None:
            config["stop_iter"] = stop_iter_override
        else:
            config["stop_iter"] = max(config.get("stop_iter", total_iterations), total_iterations)
        config["stop_iter"] = max(config["stop_iter"], config.get("start_iter", 1))
        if "occupancy_mode" not in config:
            config["occupancy_mode"] = "occupancy_shift"
        self.mesh_config = config
        return config

    def _check_gaussian_numerics(self, label: str) -> None:
        """Detect NaNs/Infs or extreme magnitudes before hitting CUDA kernels."""
        stats = {
            "xyz": self.gaussians.get_xyz,
            "scaling": self.gaussians.get_scaling,
            "rotation": self.gaussians.get_rotation,
            "opacity": self.gaussians.get_opacity,
        }
        for name, tensor in stats.items():
            if not torch.isfinite(tensor).all():
                invalid_mask = ~torch.isfinite(tensor)
                num_bad = int(invalid_mask.sum().item())
                example_idx = invalid_mask.nonzero(as_tuple=False)[:5].flatten().tolist()
                raise RuntimeError(
                    f"[NUMERIC] Detected {num_bad} non-finite entries in '{name}' "
                    f"during {label}. Sample indices: {example_idx}"
                )
            max_abs = tensor.abs().max().item()
            if max_abs > 1e6:
                print(
                    f"[WARN] Large magnitude detected in '{name}' during {label}: "
                    f"{max_abs:.3e}"
                )

    def _warmup_mesh_visibility(self) -> None:
        warmup_views = self.scene.getTrainCameras_warn_up(
            iteration=1,
            warn_until_iter=self.args.warn_until_iter,
            scale=1.0,
            scale2=2.0,
        )
        for view in warmup_views:
            render_simp(
                view,
                self.gaussians,
                self.pipe,
                self.background,
                culling=self.gaussians._culling[:, view.uid],
            )

    def _select_view(self) -> int:
        if self.fixed_view_idx is not None:
            return self.fixed_view_idx
        if not self.pending_indices:
            self.pending_indices = list(range(len(self.cameras)))
            random.shuffle(self.pending_indices)
        return self.pending_indices.pop()

    def _log_iteration(
        self,
        iteration: int,
        view_idx: int,
        total_loss: float,
        depth_loss: float,
        mesh_loss: float,
        depth_mae: float,
        valid_fraction: float,
        valid_pixels: int,
        extra: Dict[str, float],
    ) -> None:
        record = {
            "iteration": iteration,
            "view_index": view_idx,
            "total_loss": total_loss,
            "depth_loss": depth_loss,
            "mesh_loss": mesh_loss,
            "depth_mae": depth_mae,
            "valid_fraction": valid_fraction,
            "valid_pixels": valid_pixels,
        }
        if self.ema_depth is not None:
            record["ema_depth_loss"] = self.ema_depth
        if self.ema_mesh is not None:
            record["ema_mesh_loss"] = self.ema_mesh
        record.update(extra)
        with open(self.loss_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.loss_log_path, "w", encoding="utf-8"):
            pass

        for iteration in range(1, self.args.iterations + 1):
            self.gaussians.update_learning_rate(iteration)
            view_idx = self._select_view()
            viewpoint = self.cameras[view_idx]

            depth_record = self.depth_provider.get(view_idx, self.device)
            render_pkg = self.render_view(viewpoint)
            pred_depth = render_pkg["median_depth"]

            depth_loss, depth_mae, valid_fraction, valid_pixels = compute_depth_loss(
                predicted=pred_depth,
                target=depth_record.depth,
                mask=depth_record.valid_mask,
                epsilon=self.args.depth_loss_epsilon,
            )
            if valid_pixels == 0:
                if iteration % self.args.log_interval == 0 or iteration == 1:
                    print(f"[Iter {iteration:05d}] skip view {view_idx} (no valid depth)")
                continue

            mask_valid = depth_record.valid_mask.to(pred_depth.device) > 0.5
            pred_valid = pred_depth[mask_valid]
            target_valid = depth_record.depth[mask_valid]

            if not self.printed_depth_diag:
                if target_valid.numel() > 0 and pred_valid.numel() > 0:
                    print(
                        "[DIAG] First depth batch — target range {t_min:.4f} – {t_max:.4f}, "
                        "predicted range {p_min:.4f} – {p_max:.4f}".format(
                            t_min=float(target_valid.min().item()),
                            t_max=float(target_valid.max().item()),
                            p_min=float(pred_valid.min().item()),
                            p_max=float(pred_valid.max().item()),
                        )
                    )
                self.printed_depth_diag = True

            total_loss = depth_loss
            mesh_loss_tensor = torch.zeros_like(depth_loss)
            mesh_pkg: Dict[str, torch.Tensor] = {}
            mesh_active = self.mesh_enabled and iteration >= self.mesh_config.get("start_iter", 1)
            if mesh_active:
                self._check_gaussian_numerics(f"iter_{iteration}_before_mesh")
                mesh_pkg = compute_mesh_regularization(
                    iteration=iteration,
                    render_pkg=render_pkg,
                    viewpoint_cam=viewpoint,
                    viewpoint_idx=view_idx,
                    gaussians=self.gaussians,
                    scene=self.scene,
                    pipe=self.pipe,
                    background=self.background,
                    kernel_size=0.0,
                    config=self.mesh_config,
                    mesh_renderer=self.mesh_renderer,
                    mesh_state=self.mesh_state,
                    render_func=self.render_for_mesh,
                    weight_adjustment=100.0 / max(self.args.iterations, 1),
                    args=self.runtime_args,
                    integrate_func=integrate,
                )
                mesh_loss_tensor = mesh_pkg["mesh_loss"]
                self.mesh_state = mesh_pkg["updated_state"]
                total_loss = total_loss + mesh_loss_tensor

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if self.args.grad_clip_norm > 0.0:
                params: List[torch.Tensor] = []
                for group in self.optimizer.param_groups:
                    for param in group.get("params", []):
                        if isinstance(param, torch.Tensor) and param.requires_grad:
                            params.append(param)
                if params:
                    clip_grad_norm_(params, self.args.grad_clip_norm)

            visibility = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            self.optimizer.step(visibility, radii.shape[0])

            total_val = float(total_loss.item())
            depth_val = float(depth_loss.item())
            mesh_val = float(mesh_loss_tensor.item())
            self.ema_depth = depth_val if self.ema_depth is None else (0.9 * self.ema_depth + 0.1 * depth_val)
            self.ema_mesh = mesh_val if self.ema_mesh is None else (0.9 * self.ema_mesh + 0.1 * mesh_val)

            extra = {k: float(v.item()) for k, v in mesh_pkg.items() if hasattr(v, "item") and k.endswith("_loss")}
            if self.log_depth_stats and target_valid.numel() > 0:
                extra.update(
                    {
                        "pred_depth_min": float(pred_valid.min().item()),
                        "pred_depth_max": float(pred_valid.max().item()),
                        "pred_depth_mean": float(pred_valid.mean().item()),
                        "pred_depth_std": float(pred_valid.std(unbiased=False).item()),
                        "target_depth_min": float(target_valid.min().item()),
                        "target_depth_max": float(target_valid.max().item()),
                        "target_depth_mean": float(target_valid.mean().item()),
                        "target_depth_std": float(target_valid.std(unbiased=False).item()),
                    }
                )
            self._log_iteration(
                iteration=iteration,
                view_idx=view_idx,
                total_loss=total_val,
                depth_loss=depth_val,
                mesh_loss=mesh_val,
                depth_mae=depth_mae,
                valid_fraction=valid_fraction,
                valid_pixels=valid_pixels,
                extra=extra,
            )

            if iteration % self.args.log_interval == 0 or iteration == 1:
                print(
                    "[Iter {iter:05d}] loss={loss:.6f} depth={depth:.6f} mesh={mesh:.6f} "
                    "mae={mae:.6f} valid={valid:.3f}".format(
                        iter=iteration,
                        loss=total_val,
                        depth=depth_val,
                        mesh=mesh_val,
                        mae=depth_mae,
                        valid=valid_fraction,
                    )
                )

            if mesh_active and mesh_pkg.get("gaussians_changed", False):
                self.mesh_state = reset_mesh_state_at_next_iteration(self.mesh_state)

            if self.args.export_interval > 0 and iteration % self.args.export_interval == 0:
                self._export_state(iteration)

        self._export_state(self.args.iterations, final=True)

    def _sink_path(self, iteration: int, final: bool = False) -> Path:

        target_dir = self.output_dir / ("final" if final else f"iter_{iteration:05d}")
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def _export_state(self, iteration: int, final: bool = False) -> None:
        target_dir = self._sink_path(iteration, final)
        ply_path = target_dir / f"gaussians_iter_{iteration:05d}.ply"
        save_mesh = (
            self.mesh_enabled
            and (iteration >= self.mesh_config.get("start_iter", 1) or final)
            and self.mesh_state
        )
        if save_mesh and self.mesh_state.get("delaunay_tets") is not None:
            mesh_path = target_dir / f"mesh_iter_{iteration:05d}.ply"
            export_mesh_from_gaussians(
                gaussians=self.gaussians,
                mesh_state=self.mesh_state,
                output_path=str(mesh_path),
                reference_camera=None,
            )
        self.gaussians.save_ply(str(ply_path))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth-supervised training for Gaussian Splatting.")
    parser.add_argument("--ply_path", type=str, required=True, help="Initial Gaussian PLY file.")
    parser.add_argument("--camera_poses", type=str, required=True, help="Camera pose JSON compatible with ply2mesh.load_cameras_from_json.")
    parser.add_argument("--depth_dir", type=str, required=True, help="Folder of per-view depth .npy files.")
    parser.add_argument("--output_dir", type=str, default="./depth_training_output", help="Directory for logs and exports.")
    parser.add_argument("--iterations", type=int, default=5000, help="Number of optimization steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device identifier.")
    parser.add_argument("--data_device", type=str, default="cpu", help="Device to store camera image tensors.")
    parser.add_argument("--image_width", type=int, default=1280, help="Rendered width.")
    parser.add_argument("--image_height", type=int, default=720, help="Rendered height.")
    parser.add_argument("--fov_y", type=float, default=75.0, help="Vertical field of view in degrees.")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Scale factor applied to loaded depth maps.")
    parser.add_argument("--depth_offset", type=float, default=0.0, help="Additive offset applied to depth.")
    parser.add_argument("--depth_clip_min", type=float, default=None, help="Minimum depth after scaling.")
    parser.add_argument("--depth_clip_max", type=float, default=None, help="Maximum depth after scaling.")
    parser.add_argument("--depth_loss_epsilon", type=float, default=1e-6, help="Stability epsilon for depth loss denominator.")
    parser.add_argument("--mesh_config", type=str, default="medium", help="Mesh-in-the-loop configuration name.")
    parser.add_argument("--mesh_start_iter", type=int, default=1, help="Iteration to start mesh regularization.")
    parser.add_argument("--mesh_stop_iter", type=int, default=None, help="Iteration to stop mesh regularization.")
    parser.add_argument("--export_interval", type=int, default=1000, help="Export mesh/ply every N iterations.")
    parser.add_argument("--log_interval", type=int, default=100, help="Console log interval.")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0, help="Gradient clipping norm (0 disables).")
    parser.add_argument("--initial_lr_scale", type=float, default=1.0, help="Scaling factor for position learning rate.")
    parser.add_argument("--convert_shs_python", action="store_true", help="Use PyTorch SH conversion (debug only).")
    parser.add_argument("--compute_cov3d_python", action="store_true", help="Use PyTorch covariance (debug only).")
    parser.add_argument("--debug", action="store_true", help="Enable renderer debug outputs.")
    parser.add_argument("--disable_mip_filter", action="store_true", help="Disable 3D Mip filter.")
    parser.add_argument("--sh_degree", type=int, default=0, help="Spherical harmonic degree for Gaussian colors.")
    parser.add_argument("--mesh_regularization", action="store_true", help="Enable mesh-in-the-loop regularization.")
    parser.add_argument("--freeze_colors", dest="freeze_colors", action="store_true", help="Freeze SH features during depth training.", default=True)
    parser.add_argument("--no-freeze_colors", dest="freeze_colors", action="store_false", help="Allow SH features to be optimized.")
    parser.add_argument("--warn_until_iter", type=int, default=3000, help="Warmup iterations for densification/mesh utilities.")
    parser.add_argument("--imp_metric", type=str, default="outdoor", choices=["outdoor", "indoor"], help="Importance metric for mesh sampling heuristics.")
    parser.add_argument("--depth_reinit_iter", type=int, default=2000, help="Iteration to trigger optional depth reinitialization routines.")
    parser.add_argument("--fixed_view_idx", type=int, default=None, help="If provided, always train on this camera index (for debugging).")
    parser.add_argument("--log_depth_stats", action="store_true", help="Record detailed depth statistics per iteration.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    safe_state(False)
    trainer = DepthTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
