#!/usr/bin/env python3
"""Filter 3D Gaussian Splatting models based on voxelized statistics.

The script keeps voxels that are either sufficiently dense or contain very
fine Gaussians, then optionally expands the kept region by a halo and writes a
trimmed PLY. It follows the core workflow outlined in the clean_3dgs design
document.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source 3DGS PLY file")
    parser.add_argument(
        "--output",
        help=(
            "Path to the cleaned PLY file. Defaults to '<input>_cleaned.ply' next "
            "to the source file."
        ),
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.25,
        help="Voxel edge length used for statistics (default: 0.25 in world units)",
    )
    parser.add_argument(
        "--density-keep-ratio",
        type=float,
        default=0.3,
        help="Fraction of the high-density reference used as the keep threshold",
    )
    parser.add_argument(
        "--density-reference-percentile",
        type=float,
        default=0.10,
        help="Top percentile of voxel densities used to build the reference (default: 10%)",
    )
    parser.add_argument(
        "--volume-keep-ratio",
        type=float,
        default=3.0,
        help="Multiplier applied to the fine-volume reference for the detail guard",
    )
    parser.add_argument(
        "--volume-reference-percentile",
        type=float,
        default=0.15,
        help="Bottom percentile of voxel volumes used as the fine detail reference",
    )
    parser.add_argument(
        "--halo-voxels",
        type=int,
        default=3,
        help="Voxel radius for spatial halo expansion (default: 3)",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        help="Optional cap on the number of Gaussians to keep after filtering",
    )
    parser.add_argument(
        "--disable-component-filter",
        action="store_true",
        help="Keep all disconnected voxel clusters instead of retaining only the largest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when downsampling to --max-gaussians",
    )
    args = parser.parse_args()

    if args.voxel_size <= 0:
        parser.error("--voxel-size must be positive")
    if args.density_keep_ratio <= 0:
        parser.error("--density-keep-ratio must be positive")
    if not 0 < args.density_reference_percentile <= 1:
        parser.error("--density-reference-percentile must be in (0, 1]")
    if args.volume_keep_ratio <= 0:
        parser.error("--volume-keep-ratio must be positive")
    if not 0 < args.volume_reference_percentile <= 1:
        parser.error("--volume-reference-percentile must be in (0, 1]")
    if args.halo_voxels < 0:
        parser.error("--halo-voxels must be non-negative")
    if args.max_gaussians is not None and args.max_gaussians <= 0:
        parser.error("--max-gaussians must be a positive integer when provided")

    return args


def extract_positions(vertex_data: np.ndarray) -> np.ndarray:
    required = ("x", "y", "z")
    missing = [field for field in required if field not in vertex_data.dtype.names]
    if missing:
        raise ValueError(f"Input PLY is missing required position fields: {missing}")

    positions = np.empty((vertex_data.shape[0], 3), dtype=np.float64)
    positions[:, 0] = vertex_data["x"]
    positions[:, 1] = vertex_data["y"]
    positions[:, 2] = vertex_data["z"]
    return positions


def extract_log_scales(vertex_data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    scale_names = [name for name in vertex_data.dtype.names if name.startswith("scale_")]
    if not scale_names:
        raise ValueError("Input PLY is missing 'scale_*' fields required for volume estimates")
    try:
        scale_names.sort(key=lambda name: int(name.split("_")[-1]))
    except ValueError as exc:
        raise ValueError("Scale fields must follow the 'scale_<index>' naming convention") from exc

    log_scales = np.empty((vertex_data.shape[0], len(scale_names)), dtype=np.float64)
    for idx, field in enumerate(scale_names):
        log_scales[:, idx] = vertex_data[field]
    return log_scales, scale_names


def compute_reference(values: np.ndarray, fraction: float, descending: bool) -> float:
    if values.size == 0:
        return float("nan")
    sorted_vals = np.sort(values)
    if descending:
        sorted_vals = sorted_vals[::-1]
    top_n = max(int(math.ceil(len(sorted_vals) * fraction)), 1)
    subset = sorted_vals[:top_n]
    return float(np.median(subset))


def describe_distribution(values: np.ndarray, label: str, fmt: str) -> None:
    if values.size == 0:
        print(f"[STATS] {label}: no data")
        return
    quantiles = np.percentile(values, [0, 50, 95, 100])
    summary = " ".join(
        (
            f"min={format(quantiles[0], fmt)}",
            f"median={format(quantiles[1], fmt)}",
            f"p95={format(quantiles[2], fmt)}",
            f"max={format(quantiles[3], fmt)}",
        )
    )
    print(f"[STATS] {label}: {summary}")


def build_halo_offsets(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros((0, 3), dtype=np.int32)
    coords = np.arange(-radius, radius + 1, dtype=np.int32)
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    offsets = grid.reshape(-1, 3)
    keep = np.any(offsets != 0, axis=1)
    return offsets[keep]


NEIGHBOR_OFFSETS = np.array(
    [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ],
    dtype=np.int32,
)


def filter_largest_component(
    keep_mask: np.ndarray,
    unique_voxels: np.ndarray,
    voxel_to_index: dict[tuple[int, int, int], int],
    neighbor_offsets: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    visited = np.zeros_like(keep_mask, dtype=bool)
    largest_component: list[int] = []
    largest_size = 0
    component_count = 0

    for start_idx in np.flatnonzero(keep_mask):
        if visited[start_idx]:
            continue
        component_count += 1
        queue: deque[int] = deque([int(start_idx)])
        visited[start_idx] = True
        current_component: list[int] = []

        while queue:
            idx = queue.popleft()
            current_component.append(idx)
            base = unique_voxels[idx]
            for offset in neighbor_offsets:
                neighbor_coord = tuple(int(v) for v in base + offset)
                neighbor_idx = voxel_to_index.get(neighbor_coord)
                if neighbor_idx is None or visited[neighbor_idx] or not keep_mask[neighbor_idx]:
                    continue
                visited[neighbor_idx] = True
                queue.append(neighbor_idx)

        if len(current_component) > largest_size:
            largest_size = len(current_component)
            largest_component = current_component

    new_keep = np.zeros_like(keep_mask, dtype=bool)
    if largest_component:
        new_keep[np.array(largest_component, dtype=np.int64)] = True
    return new_keep, component_count, largest_size


def format_vec(values: Iterable[float]) -> str:
    return ", ".join(f"{float(v):.3f}" for v in values)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")

    print(f"[INFO] Loading PLY: {input_path}")
    ply = PlyData.read(str(input_path))
    if "vertex" not in ply:
        raise ValueError("Input PLY does not contain a 'vertex' element")

    vertex_data = np.asarray(ply["vertex"].data)
    total_gaussians = vertex_data.shape[0]
    print(f"[INFO] Total Gaussians: {total_gaussians}")
    if total_gaussians == 0:
        raise ValueError("Input PLY contains no Gaussians to process")

    positions = extract_positions(vertex_data)
    log_scales, _ = extract_log_scales(vertex_data)
    # 3DGS stores log-scale radii; exponentiate to obtain axis lengths and volume.
    volumes = np.exp(np.sum(log_scales, axis=1))

    bounds_min = positions.min(axis=0)
    bounds_max = positions.max(axis=0)
    print(f"[INFO] Bounding box min: [{format_vec(bounds_min)}]")
    print(f"[INFO] Bounding box max: [{format_vec(bounds_max)}]")

    voxel_indices = np.floor((positions - bounds_min) / args.voxel_size).astype(np.int32)
    unique_voxels, inverse_indices, counts = np.unique(
        voxel_indices, axis=0, return_inverse=True, return_counts=True
    )
    voxel_count = unique_voxels.shape[0]
    print(f"[INFO] Occupied voxels: {voxel_count}")

    voxel_to_index = {
        tuple(int(v) for v in coord): idx for idx, coord in enumerate(unique_voxels)
    }

    volume_sums = np.bincount(inverse_indices, weights=volumes, minlength=voxel_count)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_volumes = volume_sums / counts

    describe_distribution(counts.astype(np.float64), "Voxel density", ".0f")
    describe_distribution(avg_volumes, "Voxel avg_volume", ".3e")

    count_ref = compute_reference(
        counts.astype(np.float64), args.density_reference_percentile, descending=True
    )
    count_threshold = args.density_keep_ratio * count_ref

    volume_ref = compute_reference(
        avg_volumes, args.volume_reference_percentile, descending=False
    )
    if not np.isfinite(volume_ref) or volume_ref <= 0:
        volume_ref = float(np.median(avg_volumes))
    volume_threshold = args.volume_keep_ratio * volume_ref

    print(
        f"[INFO] Density ref={count_ref:.3f}, threshold={count_threshold:.3f} (ratio={args.density_keep_ratio})"
    )
    print(
        f"[INFO] Volume ref={volume_ref:.6e}, threshold={volume_threshold:.6e} (ratio={args.volume_keep_ratio})"
    )

    keep_by_density = counts >= count_threshold
    keep_by_detail = (counts < count_threshold) & (avg_volumes <= volume_threshold)
    keep_raw = keep_by_density | keep_by_detail

    kept_voxels_raw = int(np.count_nonzero(keep_raw))
    print(
        f"[INFO] Voxels kept before halo: {kept_voxels_raw} ({kept_voxels_raw / voxel_count:.2%})"
    )

    keep_final = keep_raw.copy()
    if args.halo_voxels > 0 and np.any(keep_raw):
        offsets = build_halo_offsets(args.halo_voxels)
        if offsets.size:
            for idx in np.flatnonzero(keep_raw):
                base = unique_voxels[idx]
                for offset in offsets:
                    neighbor = tuple(int(v) for v in base + offset)
                    neighbor_idx = voxel_to_index.get(neighbor)
                    if neighbor_idx is not None:
                        keep_final[neighbor_idx] = True

    kept_voxels_final = int(np.count_nonzero(keep_final))
    print(
        f"[INFO] Voxels kept after halo: {kept_voxels_final} ({kept_voxels_final / voxel_count:.2%})"
    )

    if not args.disable_component_filter and np.any(keep_final):
        keep_final, component_count, largest_size = filter_largest_component(
            keep_final, unique_voxels, voxel_to_index, NEIGHBOR_OFFSETS
        )
        print(
            f"[INFO] Component filter kept 1 of {component_count} components ({largest_size} voxels)"
        )
    elif args.disable_component_filter:
        print("[INFO] Component filter disabled; keeping all voxel clusters")

    keep_mask = keep_final[inverse_indices]
    kept_gaussians = int(np.count_nonzero(keep_mask))
    print(f"[INFO] Gaussians kept after voxel filtering: {kept_gaussians}")

    if kept_gaussians == 0:
        raise ValueError("Filtering removed all Gaussians; consider relaxing thresholds")

    if args.max_gaussians is not None and kept_gaussians > args.max_gaussians:
        rng = np.random.default_rng(args.seed)
        kept_indices = np.flatnonzero(keep_mask)
        sample = rng.choice(kept_indices, size=args.max_gaussians, replace=False)
        sample.sort()
        new_keep_mask = np.zeros_like(keep_mask, dtype=bool)
        new_keep_mask[sample] = True
        keep_mask = new_keep_mask
        kept_gaussians = args.max_gaussians
        print(f"[INFO] Downsampled to max_gaussians={args.max_gaussians}")

    filtered_vertices = np.asarray(vertex_data[keep_mask], dtype=vertex_data.dtype)

    if output_path == input_path:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        if not backup_path.exists():
            print(f"[INFO] Creating backup at {backup_path}")
            shutil.copyfile(input_path, backup_path)
        else:
            print(f"[WARNING] Backup already exists at {backup_path}; reusing it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertex_element = PlyElement.describe(filtered_vertices, "vertex")
    new_ply = PlyData([vertex_element], text=ply.text, byte_order=ply.byte_order)
    new_ply.comments = list(ply.comments)
    print(f"[INFO] Writing cleaned PLY to {output_path}")
    new_ply.write(str(output_path))
    print(f"[INFO] Done. Final Gaussians: {kept_gaussians}")


if __name__ == "__main__":
    main()
