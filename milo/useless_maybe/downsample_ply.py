#!/usr/bin/env python3
"""Downsample a COLMAP point cloud PLY file to a target number of points.

This keeps the original vertex properties intact and preserves the binary
endianness of the source file. Usage example:

    python downsample_ply.py \
        --input data/Bridge/sparse/0/points3D.ply \
        --output data/Bridge/sparse/0/points3D_downsampled.ply \
        --target 4000000

If the input already contains fewer points than the target, the script simply
copies it to the destination (or leaves it unchanged when writing in-place).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source PLY file")
    parser.add_argument(
        "--output",
        help=(
            "Path to the output PLY file. If omitted, the input is overwritten "
            "after creating a backup with suffix .bak"
        ),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=4_000_000,
        help="Desired maximum number of points (default: 4,000,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")

    print(f"[INFO] Loading PLY: {input_path}")
    ply = PlyData.read(str(input_path))

    if "vertex" not in ply:
        raise ValueError("Input PLY does not contain a vertex element")

    vertex_data = ply["vertex"]
    total_vertices = len(vertex_data)
    target = max(int(args.target), 0)
    print(f"[INFO] Total vertices: {total_vertices}")
    print(f"[INFO] Target vertices: {target}")

    if target == 0 or total_vertices <= target:
        print("[INFO] No downsampling needed.")
        if output_path != input_path:
            print(f"[INFO] Copying file to {output_path}")
            shutil.copyfile(input_path, output_path)
        else:
            print("[INFO] Input already satisfies target; nothing to do.")
        return

    rng = np.random.default_rng(args.seed)
    print("[INFO] Sampling indices...")
    sample_indices = rng.choice(total_vertices, size=target, replace=False)
    sample_indices.sort()
    downsampled_vertex = vertex_data[sample_indices]

    print("[INFO] Preparing PLY structure...")
    new_vertex_element = PlyElement.describe(downsampled_vertex, "vertex")
    new_ply = PlyData([new_vertex_element], text=ply.text, byte_order=ply.byte_order)
    new_ply.comments = ply.comments

    if output_path == input_path:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        if not backup_path.exists():
            print(f"[INFO] Creating backup at {backup_path}")
            shutil.copyfile(input_path, backup_path)
        else:
            print(f"[WARNING] Backup already exists at {backup_path}; reusing it.")

    os.makedirs(output_path.parent, exist_ok=True)
    print(f"[INFO] Writing downsampled PLY to {output_path}")
    new_ply.write(str(output_path))
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
