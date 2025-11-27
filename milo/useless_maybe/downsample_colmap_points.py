#!/usr/bin/env python3
"""Downsample COLMAP points3D.bin (and optionally regenerate points3D.ply)."""

from __future__ import annotations

import argparse
import os
import shutil
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-bin",
        required=True,
        help="Path to the COLMAP points3D.bin file",
    )
    parser.add_argument(
        "--output-bin",
        help=(
            "Path to the output points3D.bin file. Defaults to overwriting the input "
            "after creating a .bak backup."
        ),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=4_000_000,
        help="Maximum number of points to keep (default: 4,000,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--mode",
        choices=("random", "radius"),
        default="random",
        help=(
            "Sampling strategy: 'random' selects uniformly at random; "
            "'radius' keeps points closest to the centroid (default: random)"
        ),
    )
    parser.add_argument(
        "--ply-output",
        help=(
            "Optional path to write a matching points3D.ply. Defaults to the sibling "
            "points3D.ply next to the binary file when omitted. Use '--ply-output '' ' to skip."
        ),
    )
    return parser.parse_args()


RECORD_STRUCT = struct.Struct("<QdddBBBd")
TRACK_LEN_STRUCT = struct.Struct("<Q")


def collect_positions(path: Path, num_points: int) -> np.ndarray:
    """Read point positions once to compute spatial statistics."""
    positions = np.empty((num_points, 3), dtype=np.float32)
    with path.open("rb") as fin:
        fin.seek(8)  # Skip count header
        for idx in range(num_points):
            record_bytes = fin.read(RECORD_STRUCT.size)
            if not record_bytes:
                raise EOFError("Unexpected end of file while gathering positions")
            _, x, y, z, _, _, _, _ = RECORD_STRUCT.unpack(record_bytes)
            positions[idx] = (x, y, z)
            (track_len,) = TRACK_LEN_STRUCT.unpack(fin.read(TRACK_LEN_STRUCT.size))
            fin.seek(8 * track_len, os.SEEK_CUR)
    return positions


def pick_indices(
    total: int,
    target: int,
    rng: np.random.Generator,
    mode: str,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    if target <= 0 or total <= target:
        return np.arange(total, dtype=np.int64)
    if mode == "random":
        indices = rng.choice(total, size=target, replace=False)
    elif mode == "radius":
        if positions is None:
            raise ValueError("Positions are required for radius-based sampling.")
        centroid = positions.mean(axis=0, dtype=np.float64)
        dists = np.sum((positions - centroid) ** 2, axis=1)
        partition = np.argpartition(dists, target - 1)[:target]
        indices = np.sort(partition)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
    if not np.all(np.diff(indices) >= 0):
        indices.sort()
    return indices.astype(np.int64)


def write_ply(path: Path, positions: np.ndarray, colors: np.ndarray) -> None:
    count = positions.shape[0]
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.empty(count, dtype=dtype)
    elements["x"] = positions[:, 0]
    elements["y"] = positions[:, 1]
    elements["z"] = positions[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["red"] = colors[:, 0]
    elements["green"] = colors[:, 1]
    elements["blue"] = colors[:, 2]
    ply = PlyData([PlyElement.describe(elements, "vertex")], text=False)
    ply.write(str(path))


def main() -> None:
    args = parse_args()
    input_bin = Path(args.input_bin).expanduser().resolve()
    output_bin = Path(args.output_bin).expanduser().resolve() if args.output_bin else input_bin

    if not input_bin.exists():
        raise FileNotFoundError(f"Input binary file not found: {input_bin}")

    rng = np.random.default_rng(args.seed)

    with input_bin.open("rb") as fin:
        num_points = struct.unpack("<Q", fin.read(8))[0]
    print(f"[INFO] Input points: {num_points}")

    target = max(int(args.target), 0)
    positions_all = None
    if args.mode == "radius" and target < num_points:
        print("[INFO] Gathering positions for radius-based sampling...")
        positions_all = collect_positions(input_bin, num_points)
    keep_indices = pick_indices(num_points, target, rng, args.mode, positions_all)
    keep_count = keep_indices.shape[0]
    print(f"[INFO] Keeping {keep_count} points.")

    if output_bin == input_bin:
        backup = input_bin.with_suffix(input_bin.suffix + ".bak")
        if not backup.exists():
            print(f"[INFO] Creating backup at {backup}")
            shutil.copyfile(input_bin, backup)
        else:
            print(f"[WARNING] Backup already exists at {backup}; reusing it.")
        tmp_bin = input_bin.with_suffix(input_bin.suffix + ".tmp")
    else:
        tmp_bin = output_bin
        os.makedirs(tmp_bin.parent, exist_ok=True)

    positions = np.empty((keep_count, 3), dtype=np.float32)
    colors = np.empty((keep_count, 3), dtype=np.uint8)

    keep_ptr = 0
    next_keep: Optional[int] = int(keep_indices[keep_ptr]) if keep_count > 0 else None

    with input_bin.open("rb") as fin, tmp_bin.open("wb") as fout:
        fin.seek(8)
        fout.write(struct.pack("<Q", keep_count))

        for idx in range(num_points):
            record_bytes = fin.read(RECORD_STRUCT.size)
            if not record_bytes:
                raise EOFError("Unexpected end of file while reading record")
            point_id, x, y, z, r, g, b, error = RECORD_STRUCT.unpack(record_bytes)
            track_len_bytes = fin.read(TRACK_LEN_STRUCT.size)
            if not track_len_bytes:
                raise EOFError("Unexpected end of file while reading track length")
            (track_len,) = TRACK_LEN_STRUCT.unpack(track_len_bytes)
            track_bytes = fin.read(8 * track_len)
            if len(track_bytes) != 8 * track_len:
                raise EOFError("Unexpected end of file while reading track entries")

            if next_keep is not None and idx == next_keep:
                fout.write(record_bytes)
                fout.write(track_len_bytes)
                fout.write(track_bytes)
                positions[keep_ptr] = (x, y, z)
                colors[keep_ptr] = (r, g, b)
                keep_ptr += 1
                if keep_ptr == keep_count:
                    next_keep = None
                    break
                else:
                    next_keep = int(keep_indices[keep_ptr])

        if keep_ptr != keep_count:
            # If we broke early but still have to consume the rest to keep file position consistent
            for idx in range(idx + 1, num_points):
                record_bytes = fin.read(RECORD_STRUCT.size)
                if not record_bytes:
                    break
                track_len_bytes = fin.read(TRACK_LEN_STRUCT.size)
                if not track_len_bytes:
                    break
                (track_len,) = TRACK_LEN_STRUCT.unpack(track_len_bytes)
                fin.seek(8 * track_len, os.SEEK_CUR)

    if output_bin == input_bin:
        os.replace(tmp_bin, input_bin)
    elif tmp_bin != output_bin:
        os.replace(tmp_bin, output_bin)

    if args.ply_output is not None:
        ply_path = Path(args.ply_output).expanduser().resolve()
        if ply_path == Path(""):
            print("[INFO] Skipping PLY generation.")
        else:
            os.makedirs(ply_path.parent, exist_ok=True)
            print(f"[INFO] Writing PLY to {ply_path}")
            write_ply(ply_path, positions, colors)
    else:
        ply_path = input_bin.with_name("points3D.ply")
        print(f"[INFO] Writing PLY to {ply_path}")
        write_ply(ply_path, positions, colors)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
