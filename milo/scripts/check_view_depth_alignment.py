#!/usr/bin/env python3
"""
Utility to inspect specific Discoverse camera views and verify whether their
depth maps stay consistent with the reference bridge mesh.

Example:
    python scripts/check_view_depth_alignment.py \\
        --camera_json milo/data/bridge_clean/camera_poses_cam1.json \\
        --depth_dir milo/data/bridge_clean/depth \\
        --ply_path milo/data/bridge_clean/yufu_bridge_cleaned.ply \\
        --view_indices 72 187 --stride 8 --max_samples 20000
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

OPENGL_TO_COLMAP = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


@dataclass
class CameraPose:
    name: str
    rotation_c2w: np.ndarray  # (3, 3)
    camera_center: np.ndarray  # (3,)


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Quaternion needs 4 numbers, got shape {q.shape}")
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
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_camera_list(json_path: Path) -> List[CameraPose]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict):
        for key in ("frames", "poses", "camera_poses"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
        else:
            raise ValueError(f"{json_path} does not contain a pose list.")

    cameras: List[CameraPose] = []
    for idx, entry in enumerate(payload):
        if "quaternion" in entry:
            rotation_c2w = quaternion_to_rotation_matrix(entry["quaternion"])
        elif "rotation" in entry:
            rotation_c2w = np.asarray(entry["rotation"], dtype=np.float64)
        else:
            raise KeyError(f"Entry {idx} missing quaternion/rotation.")

        rotation_c2w = rotation_c2w @ OPENGL_TO_COLMAP

        if "position" in entry:
            camera_center = np.asarray(entry["position"], dtype=np.float64)
        elif "translation" in entry:
            camera_center = np.asarray(entry["translation"], dtype=np.float64)
        else:
            raise KeyError(f"Entry {idx} missing position/translation.")

        name = entry.get("name") or f"view_{idx:04d}"
        cameras.append(CameraPose(name=name, rotation_c2w=rotation_c2w, camera_center=camera_center))
    return cameras


def build_intrinsics(width: int, height: int, fov_y_deg: float) -> Dict[str, float]:
    fov_y = math.radians(fov_y_deg)
    fy = 0.5 * height / math.tan(0.5 * fov_y)
    aspect = width / height
    fov_x = 2.0 * math.atan(aspect * math.tan(fov_y * 0.5))
    fx = 0.5 * width / math.tan(0.5 * fov_x)
    return {
        "fx": fx,
        "fy": fy,
        "cx": (width - 1) * 0.5,
        "cy": (height - 1) * 0.5,
    }


def depth_to_world_points(
    depth_map: np.ndarray,
    camera: CameraPose,
    intrinsics: Dict[str, float],
    stride: int,
) -> np.ndarray:
    h, w = depth_map.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    samples = depth_map[grid_y, grid_x]
    valid = np.isfinite(samples) & (samples > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    samples = samples[valid]
    px = grid_x[valid].astype(np.float64)
    py = grid_y[valid].astype(np.float64)
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    x_cam = (px - cx) / fx * samples
    y_cam = (py - cy) / fy * samples
    z_cam = samples
    cam_points = np.stack([x_cam, y_cam, z_cam], axis=1)
    world_points = cam_points @ camera.rotation_c2w.T + camera.camera_center
    return world_points.astype(np.float64)


def summarize_points(points: np.ndarray) -> Dict[str, np.ndarray]:
    if points.size == 0:
        return {"min": np.array([]), "max": np.array([]), "mean": np.array([])}
    return {
        "min": points.min(axis=0),
        "max": points.max(axis=0),
        "mean": points.mean(axis=0),
    }


def build_mesh_kdtree(mesh_path: Path, sample_vertices: int) -> tuple[cKDTree, np.ndarray]:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        combined = []
        for geom in mesh.geometry.values():
            combined.append(np.asarray(geom.vertices, dtype=np.float64))
        vertices = np.concatenate(combined, axis=0) if combined else np.empty((0, 3), dtype=np.float64)
    else:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError(f"No vertices found in {mesh_path}")
    if sample_vertices > 0 and sample_vertices < len(vertices):
        rng = np.random.default_rng(seed=0)
        choice = rng.choice(len(vertices), size=sample_vertices, replace=False)
        vertices = vertices[choice]
    tree = cKDTree(vertices)
    return tree, vertices


def compute_distances(tree: cKDTree, points: np.ndarray, max_samples: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if max_samples > 0 and points.shape[0] > max_samples:
        rng = np.random.default_rng(seed=42)
        choice = rng.choice(points.shape[0], size=max_samples, replace=False)
        pts = points[choice]
    else:
        pts = points
    distances, _ = tree.query(pts, workers=-1)
    return distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Discoverse depth / pose consistency for given view indices.")
    parser.add_argument("--camera_json", type=Path, required=True, help="Path to camera pose JSON.")
    parser.add_argument("--depth_dir", type=Path, required=True, help="Directory containing depth_img_0_*.npy files.")
    parser.add_argument("--ply_path", type=Path, required=True, help="Reference mesh PLY used during capture.")
    parser.add_argument("--view_indices", type=int, nargs="+", required=True, help="List of view indices to inspect.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov_y", type=float, default=75.0)
    parser.add_argument("--stride", type=int, default=8, help="Pixel stride when sub-sampling depth map.")
    parser.add_argument("--mesh_sample", type=int, default=200000, help="Number of mesh vertices used to build KD-tree.")
    parser.add_argument("--max_samples", type=int, default=20000, help="Maximum number of world points for NN query.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = load_camera_list(args.camera_json)
    if max(args.view_indices) >= len(cameras):
        raise IndexError(f"Requested view index exceeds available cameras ({len(cameras)}).")
    intr = build_intrinsics(args.width, args.height, args.fov_y)
    depth_dir = args.depth_dir
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory {depth_dir} not found.")

    tree, mesh_points = build_mesh_kdtree(args.ply_path, args.mesh_sample)
    mesh_bounds = np.array([mesh_points.min(axis=0), mesh_points.max(axis=0)])

    print(f"[INFO] Loaded {len(cameras)} cameras from {args.camera_json}.")
    print(f"[INFO] Mesh bounds: min={mesh_bounds[0]}, max={mesh_bounds[1]}")

    for view_idx in args.view_indices:
        camera = cameras[view_idx]
        depth_path = depth_dir / f"depth_img_0_{view_idx:03d}.npy"
        if not depth_path.is_file():
            depth_path = depth_dir / f"depth_img_0_{view_idx}.npy"
        if not depth_path.is_file():
            print(f"[WARN] Depth file for view {view_idx} missing, skipping.")
            continue
        depth = np.load(depth_path).squeeze()
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            print(f"[WARN] View {view_idx} has no valid depth pixels.")
            continue
        depth_stats = {
            "min": float(depth[valid].min()),
            "max": float(depth[valid].max()),
            "mean": float(depth[valid].mean()),
            "std": float(depth[valid].std()),
        }
        world_points = depth_to_world_points(depth, camera, intr, stride=max(1, args.stride))
        point_stats = summarize_points(world_points)
        dist = compute_distances(tree, world_points, args.max_samples)
        out_of_bounds = np.logical_or(world_points < mesh_bounds[0], world_points > mesh_bounds[1]).any(axis=1)
        print(f"\n[VIEW {view_idx:03d}] {camera.name}")
        print(f"  Depth stats   : min={depth_stats['min']:.3f} max={depth_stats['max']:.3f} "
              f"mean={depth_stats['mean']:.3f} std={depth_stats['std']:.3f}")
        if world_points.size == 0:
            print("  No valid reprojected points.")
            continue
        print(f"  World bounds  : min={point_stats['min']} max={point_stats['max']}")
        print(f"  Outside mesh BB fraction: {out_of_bounds.mean():.4f}")
        if dist.size:
            print(f"  NN distance (m): mean={dist.mean():.3f} median={np.median(dist):.3f} "
                  f"p95={np.percentile(dist,95):.3f} max={dist.max():.3f}")
        else:
            print("  NN distance: n/a (insufficient points)")


if __name__ == "__main__":
    main()
