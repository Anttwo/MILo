"""Convert meshes to their convex hull representations using trimesh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import trimesh


VALID_EXTENSIONS: Sequence[str] = (".ply", ".obj", ".stl", ".glb", ".gltf")


def iter_mesh_paths(root: Path, recursive: bool) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in VALID_EXTENSIONS:
            yield root
        return

    if not root.is_dir():
        print(f"[WARN] {root} 既不是文件也不是目录，跳过。")
        return

    pattern = "**/*" if recursive else "*"
    for path in root.glob(pattern):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def convert_to_convex_hull(
    mesh_path: Path,
    suffix: str,
    overwrite: bool,
) -> Path:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    convex = mesh.convex_hull
    output_path = mesh_path.with_name(f"{mesh_path.stem}{suffix}{mesh_path.suffix}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} 已存在，使用 --overwrite 以覆盖。")
    convex.export(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将单个 mesh 或目录中的所有 mesh 转为凸包表示，文件名追加 `_convex`。",
    )
    parser.add_argument("input_path", type=Path, help="mesh 文件或目录路径。")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="若 input_path 为目录，递归遍历所有子目录。",
    )
    parser.add_argument(
        "--suffix",
        default="_convex",
        help="输出文件名后缀（默认 _convex）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在则覆盖。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mesh_paths = list(iter_mesh_paths(args.input_path, args.recursive))
    if not mesh_paths:
        print(f"[WARN] 在 {args.input_path} 下找不到 mesh 文件（支持扩展名: {', '.join(VALID_EXTENSIONS)}）。")
        return 1

    for path in mesh_paths:
        try:
            output = convert_to_convex_hull(
                path,
                suffix=args.suffix,
                overwrite=args.overwrite,
            )
            print(f"[INFO] {path} -> {output}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] 处理 {path} 失败：{exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
