#!/usr/bin/env python3
"""
将 ScanNet 场景 (RGB-D + 相机轨迹) 转换成 Milo 训练所需的纯 COLMAP 目录。

脚本会：
1. 解析 .sens 获取 RGB 帧与 camera-to-world pose。
2. 应用 axisAlignment（若存在）以保持 ScanNet 公开场景的惯用坐标。
3. 把选中的帧原封不动写成 JPEG，生成 COLMAP cameras.txt / images.txt。
4. 从 *_vh_clean_2.ply（或备用 *_vh_clean.ply）抽取点云，写入 points3D.bin/txt。
生成结果与 milo/data/Ignatius 相同（images + sparse/0）。
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from plyfile import PlyData


@dataclass
class ScanNetFrame:
    """Minimal容器，只保留 Milo 转换需要的信息。"""

    index: int
    camera_to_world: np.ndarray  # (4, 4)
    timestamp_color: int
    timestamp_depth: int
    color_bytes: bytes


class ScanNetSensorData:
    """直接解析 ScanNet .sens（二进制 RGB-D 轨迹）。"""

    def __init__(self, sens_path: Path):
        self.sens_path = Path(sens_path)
        self._fh = None
        self.version = None
        self.sensor_name = ""
        self.intrinsic_color = None
        self.extrinsic_color = None
        self.intrinsic_depth = None
        self.extrinsic_depth = None
        self.color_compression = None
        self.depth_compression = None
        self.color_width = 0
        self.color_height = 0
        self.depth_width = 0
        self.depth_height = 0
        self.depth_shift = 0.0
        self.num_frames = 0

    def __enter__(self) -> "ScanNetSensorData":
        self._fh = self.sens_path.open("rb")
        self._read_header()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh:
            self._fh.close()
            self._fh = None

    def _read_header(self) -> None:
        fh = self._fh
        assert fh is not None
        read = fh.read
        self.version = struct.unpack("<I", read(4))[0]
        if self.version != 4:
            raise ValueError(f"Unsupported .sens version {self.version}")
        strlen = struct.unpack("<Q", read(8))[0]
        self.sensor_name = read(strlen).decode("utf-8")
        self.intrinsic_color = np.frombuffer(read(16 * 4), dtype="<f4").reshape(4, 4)
        self.extrinsic_color = np.frombuffer(read(16 * 4), dtype="<f4").reshape(4, 4)
        self.intrinsic_depth = np.frombuffer(read(16 * 4), dtype="<f4").reshape(4, 4)
        self.extrinsic_depth = np.frombuffer(read(16 * 4), dtype="<f4").reshape(4, 4)
        self.color_compression = struct.unpack("<i", read(4))[0]
        self.depth_compression = struct.unpack("<i", read(4))[0]
        self.color_width, self.color_height = struct.unpack("<II", read(8))
        self.depth_width, self.depth_height = struct.unpack("<II", read(8))
        self.depth_shift = struct.unpack("<f", read(4))[0]
        self.num_frames = struct.unpack("<Q", read(8))[0]

    def iter_frames(self) -> Iterator[ScanNetFrame]:
        if self._fh is None:
            raise RuntimeError("Sensor file is not opened. Use within a context manager.")

        for frame_idx in range(self.num_frames):
            mat = np.frombuffer(self._fh.read(16 * 4), dtype="<f4").reshape(4, 4)
            ts_color = struct.unpack("<Q", self._fh.read(8))[0]
            ts_depth = struct.unpack("<Q", self._fh.read(8))[0]
            color_size = struct.unpack("<Q", self._fh.read(8))[0]
            depth_size = struct.unpack("<Q", self._fh.read(8))[0]
            color_bytes = self._fh.read(color_size)
            _ = self._fh.read(depth_size)  # 深度暂不需要
            yield ScanNetFrame(
                index=frame_idx,
                camera_to_world=mat.astype(np.float64),
                timestamp_color=ts_color,
                timestamp_depth=ts_depth,
                color_bytes=color_bytes,
            )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """复制自 COLMAP read_write_model，实现矩阵->四元数。"""
    R = np.asarray(R, dtype=np.float64)
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array(
        [
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
        ]
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec / np.linalg.norm(qvec)


def parse_axis_alignment(meta_txt: Path) -> np.ndarray:
    """读取 axisAlignment=... 行，没有则返回单位阵。"""
    if not meta_txt.is_file():
        return np.eye(4, dtype=np.float64)

    axis = None
    with meta_txt.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip().startswith("axisAlignment"):
                values = line.split("=", 1)[1].strip().split()
                if len(values) != 16:
                    raise ValueError(f"axisAlignment 需要 16 个数，当前 {len(values)}")
                axis = np.array([float(v) for v in values], dtype=np.float64).reshape(4, 4)
                break
    if axis is None:
        axis = np.eye(4, dtype=np.float64)
    return axis


def infer_scene_id(scene_root: Path) -> str:
    """默认场景目录名就是 sceneXXXX_YY；否则尝试寻找唯一的 *.sens。"""
    if scene_root.name.startswith("scene") and "_" in scene_root.name:
        return scene_root.name
    sens_files = list(scene_root.glob("*.sens"))
    if len(sens_files) != 1:
        raise ValueError("无法唯一确定 scene id，请使用 --scene-id")
    return sens_files[0].stem


def find_point_cloud(scene_root: Path, scene_id: str, override: Optional[Path]) -> Path:
    if override:
        pc_path = Path(override)
        if not pc_path.is_file():
            raise FileNotFoundError(f"指定点云 {pc_path} 不存在")
        return pc_path
    candidates = [
        scene_root / f"{scene_id}_vh_clean_2.ply",
        scene_root / f"{scene_id}_vh_clean.ply",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError("找不到 *_vh_clean_*.ply 点云，请用 --points-source 指定")


def load_point_cloud(
    ply_path: Path,
    stride: int = 1,
    max_points: Optional[int] = None,
    seed: int = 0,
    transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(ply_path))
    verts = ply["vertex"].data
    xyz = np.vstack([verts["x"], verts["y"], verts["z"]]).T.astype(np.float64)
    if {"red", "green", "blue"}.issubset(verts.dtype.names):
        colors = np.vstack([verts["red"], verts["green"], verts["blue"]]).T.astype(np.uint8)
    else:
        colors = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)

    idx = np.arange(xyz.shape[0])
    if stride > 1:
        idx = idx[::stride]
    if max_points is not None and idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    xyz = xyz[idx]
    if transform is not None:
        if transform.shape != (4, 4):
            raise ValueError("transform must be 4x4 homogeneous matrix")
        homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
        xyz = (homo @ transform.T)[:, :3]
    return xyz, colors[idx]


def ensure_output_dirs(output_root: Path) -> Tuple[Path, Path]:
    images_dir = output_root / "images"
    sparse_dir = output_root / "sparse" / "0"
    if output_root.exists():
        existing = [p for p in output_root.iterdir() if not p.name.startswith(".")]
        if existing:
            raise FileExistsError(f"{output_root} 已存在且非空，请指定新的输出目录")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, sparse_dir


@dataclass
class ImageRecord:
    image_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    frame_index: int


def write_cameras_txt(path: Path, camera_id: int, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fh.write("# Number of cameras: 1\n")
        fh.write(f"{camera_id} PINHOLE {width} {height} {fx:.9f} {fy:.9f} {cx:.9f} {cy:.9f}\n")


def write_images_txt(path: Path, camera_id: int, records: Sequence[ImageRecord]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Image list with two lines of data per image:\n")
        fh.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fh.write(f"# Number of images: {len(records)}, mean observations per image: 0\n")
        for rec in records:
            q = rec.qvec
            t = rec.tvec
            fh.write(
                f"{rec.image_id} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} {camera_id} {rec.name}\n"
            )
            fh.write("\n")


def write_points3d_txt(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# 3D point list with one line of data per point:\n")
        fh.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        fh.write(f"# Number of points: {xyz.shape[0]}\n")
        for idx, (pos, color) in enumerate(zip(xyz, rgb), start=1):
            fh.write(
                f"{idx} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )


def write_points3d_bin(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", xyz.shape[0]))
        for idx, (pos, color) in enumerate(zip(xyz, rgb), start=1):
            fh.write(
                struct.pack(
                    "<QdddBBBd",
                    idx,
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    0.0,
                )
            )
            fh.write(struct.pack("<Q", 0))  # track length


def convert_scene(args: argparse.Namespace) -> None:
    if args.frame_step <= 0:
        raise ValueError("frame-step 必须为正整数")
    if args.start_frame < 0:
        raise ValueError("start-frame 不能为负")
    if args.points_stride <= 0:
        raise ValueError("points-stride 必须为正整数")

    scene_root = Path(args.scene_root).resolve()
    scene_id = args.scene_id or infer_scene_id(scene_root)
    sens_path = scene_root / f"{scene_id}.sens"
    if not sens_path.is_file():
        raise FileNotFoundError(f"未找到 {sens_path}")
    meta_path = scene_root / f"{scene_id}.txt"
    axis = parse_axis_alignment(meta_path) if args.apply_axis_alignment else np.eye(4, dtype=np.float64)
    images_dir, sparse_dir = ensure_output_dirs(Path(args.output).resolve())
    point_cloud_path = find_point_cloud(scene_root, scene_id, args.points_source)

    print(f"[INFO] 转换场景 {scene_id} -> {args.output}")
    print(f"[INFO] 使用点云: {point_cloud_path}")

    with ScanNetSensorData(sens_path) as sensor:
        if sensor.color_compression != 2:
            raise NotImplementedError(f"暂不支持 color_compression={sensor.color_compression} 的 .sens")

        fx = float(sensor.intrinsic_color[0, 0])
        fy = float(sensor.intrinsic_color[1, 1])
        cx = float(sensor.intrinsic_color[0, 2])
        cy = float(sensor.intrinsic_color[1, 2])
        camera_id = 1

        selected: List[ImageRecord] = []
        next_image_id = 1
        max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
        start = args.start_frame
        for frame in sensor.iter_frames():
            if frame.index < start:
                continue
            if (frame.index - start) % args.frame_step != 0:
                continue
            if max_frames is not None and len(selected) >= max_frames:
                break

            c2w = frame.camera_to_world
            if args.apply_axis_alignment:
                c2w = axis @ c2w
            if not np.all(np.isfinite(c2w)):
                print(f"[WARN] 跳过第 {frame.index} 帧：pose 含 NaN")
                continue

            w2c = np.linalg.inv(c2w)
            rot = w2c[:3, :3]
            tvec = w2c[:3, 3]
            qvec = rotmat_to_qvec(rot)
            image_name = f"frame_{frame.index:06d}.jpg"
            image_path = images_dir / image_name
            with image_path.open("wb") as im_fh:
                im_fh.write(frame.color_bytes)

            selected.append(
                ImageRecord(
                    image_id=next_image_id,
                    name=image_name,
                    qvec=qvec,
                    tvec=tvec,
                    frame_index=frame.index,
                )
            )
            next_image_id += 1
            if len(selected) % 100 == 0:
                print(f"[INFO] 已写入 {len(selected)} 张图像")

        if not selected:
            raise RuntimeError("没有任何帧满足采样条件，请检查 start/step/max 参数。")

        cams_txt = sparse_dir / "cameras.txt"
        write_cameras_txt(cams_txt, camera_id, sensor.color_width, sensor.color_height, fx, fy, cx, cy)
        imgs_txt = sparse_dir / "images.txt"
        write_images_txt(imgs_txt, camera_id, selected)

    xyz, rgb = load_point_cloud(
        point_cloud_path,
        stride=args.points_stride,
        max_points=args.points_max,
        seed=args.points_seed,
        transform=axis if args.apply_axis_alignment else None,
    )
    write_points3d_txt(sparse_dir / "points3D.txt", xyz, rgb)
    write_points3d_bin(sparse_dir / "points3D.bin", xyz, rgb)

    print(f"[INFO] 转换完成：{len(selected)} 张图像，{xyz.shape[0]} 个点。")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 ScanNet 场景转换为 Milo 所需的 COLMAP 布局。")
    parser.add_argument("--scene-root", required=True, help="包含 sceneXXXX_YY.* 的目录（目录内有 .sens/.txt/.ply）。")
    parser.add_argument("--output", required=True, help="输出目录（需不存在或为空，将创建 images/ 与 sparse/0/）。")
    parser.add_argument("--scene-id", help="可选，显式指定 sceneXXXX_YY。默认取目录名或自动推断。")
    parser.add_argument("--start-frame", type=int, default=0, help="从第多少帧开始采样（默认 0）。")
    parser.add_argument("--frame-step", type=int, default=1, help="帧采样步长，例如 5 表示每 5 帧取 1 帧。")
    parser.add_argument("--max-frames", type=int, help="最多输出多少帧，默认全部。")
    parser.add_argument("--no-axis-alignment", dest="apply_axis_alignment", action="store_false", help="不使用 axisAlignment。")
    parser.set_defaults(apply_axis_alignment=True)
    parser.add_argument("--points-source", type=Path, help="自定义点云 .ply 路径（默认自动找 *_vh_clean_2.ply）。")
    parser.add_argument("--points-stride", type=int, default=1, help="点云下采样步长（1 表示保留全部）。")
    parser.add_argument("--points-max", type=int, help="点云最多保留多少点。")
    parser.add_argument("--points-seed", type=int, default=0, help="点云随机采样用的随机种子。")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    convert_scene(args)


if __name__ == "__main__":
    main()
