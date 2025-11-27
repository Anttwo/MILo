from pathlib import Path
import math
import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import yaml

from argparse import ArgumentParser

from functional import (
    compute_delaunay_triangulation,
    extract_mesh,
    frustum_cull_mesh,
)
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_full
from gaussian_renderer.radegs import render_radegs, integrate_radegs
from arguments import PipelineParams
from regularization.regularizer.mesh import (
    initialize_mesh_regularization,
    compute_mesh_regularization,
)
from regularization.sdf.learnable import convert_occupancy_to_sdf
from utils.geometry_utils import flatten_voronoi_features

# DISCOVER-SE 相机轨迹使用 OpenGL 右手坐标系（相机前方为 -Z，向上为 +Y），
# 而 MILo/colmap 渲染管线假设的是前方 +Z、向上 -Y。需要在读入时做一次轴翻转。
OPENGL_TO_COLMAP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


class DepthProvider:
    """负责加载并缓存 Discoverse 深度图，统一形状、裁剪和掩码。"""

    def __init__(
        self,
        depth_root: Path,
        image_height: int,
        image_width: int,
        device: torch.device,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        self.depth_root = Path(depth_root)
        if not self.depth_root.is_dir():
            raise FileNotFoundError(f"深度目录不存在：{self.depth_root}")
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._cache: Dict[int, torch.Tensor] = {}
        self._mask_cache: Dict[int, torch.Tensor] = {}

    def _file_for_index(self, view_index: int) -> Path:
        return self.depth_root / f"depth_img_0_{view_index}.npy"

    def _load_numpy(self, file_path: Path) -> np.ndarray:
        depth_np = np.load(file_path)
        depth_np = np.squeeze(depth_np)
        if depth_np.ndim != 2:
            raise ValueError(f"{file_path} 深度数组维度异常：{depth_np.shape}")
        if depth_np.shape != (self.image_height, self.image_width):
            raise ValueError(
                f"{file_path} 深度分辨率应为 {(self.image_height, self.image_width)}，当前为 {depth_np.shape}"
            )
        if self.clip_min is not None or self.clip_max is not None:
            min_val = self.clip_min if self.clip_min is not None else None
            max_val = self.clip_max if self.clip_max is not None else None
            depth_np = np.clip(
                depth_np,
                min_val if min_val is not None else depth_np.min(),
                max_val if max_val is not None else depth_np.max(),
            )
        return depth_np.astype(np.float32)

    def get(self, view_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (depth_tensor, valid_mask)，均在 GPU 上。"""
        if view_index not in self._cache:
            file_path = self._file_for_index(view_index)
            if not file_path.is_file():
                raise FileNotFoundError(f"缺少深度文件：{file_path}")
            depth_np = self._load_numpy(file_path)
            depth_tensor = torch.from_numpy(depth_np).to(self.device)
            valid_mask = torch.isfinite(depth_tensor) & (depth_tensor > 0.0)
            if self.clip_min is not None:
                valid_mask &= depth_tensor >= self.clip_min
            if self.clip_max is not None:
                valid_mask &= depth_tensor <= self.clip_max
            self._cache[view_index] = depth_tensor
            self._mask_cache[view_index] = valid_mask
        return self._cache[view_index], self._mask_cache[view_index]

    def as_numpy(self, view_index: int) -> np.ndarray:
        depth_tensor, _ = self.get(view_index)
        return depth_tensor.detach().cpu().numpy()


class NormalGroundTruthCache:
    """缓存以初始高斯生成的法线 GT，避免训练阶段重复渲染。"""

    def __init__(
        self,
        cache_dir: Path,
        image_height: int,
        image_width: int,
        device: torch.device,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        self._memory_cache: Dict[int, torch.Tensor] = {}

    def _file_path(self, view_index: int) -> Path:
        return self.cache_dir / f"normal_view_{view_index:04d}.npy"

    def has(self, view_index: int) -> bool:
        return self._file_path(view_index).is_file()

    def store(self, view_index: int, normal_tensor: torch.Tensor) -> None:
        normals_np = prepare_normals(normal_tensor)
        np.save(self._file_path(view_index), normals_np.astype(np.float16))

    def get(self, view_index: int) -> torch.Tensor:
        if view_index not in self._memory_cache:
            path = self._file_path(view_index)
            if not path.is_file():
                raise FileNotFoundError(
                    f"未找到视角 {view_index} 的法线缓存：{path}，请先完成预计算。"
                )
            normals_np = np.load(path)
            expected_shape = (self.image_height, self.image_width, 3)
            if normals_np.shape != expected_shape:
                raise ValueError(
                    f"{path} 法线缓存尺寸应为 {expected_shape}，当前为 {normals_np.shape}"
                )
            normals_tensor = (
                torch.from_numpy(normals_np.astype(np.float32))
                .permute(2, 0, 1)
                .to(self.device)
            )
            self._memory_cache[view_index] = normals_tensor
        return self._memory_cache[view_index]

    def clear_memory_cache(self) -> None:
        self._memory_cache.clear()

    def ensure_all(self, cameras: Sequence[Camera], render_fn) -> None:
        total = len(cameras)
        for idx, camera in enumerate(cameras):
            if self.has(idx):
                continue
            with torch.no_grad():
                pkg = render_fn(camera)
                normal_tensor = pkg["normal"]
            self.store(idx, normal_tensor)
            if (idx + 1) % 10 == 0 or idx + 1 == total:
                print(f"[INFO] 预计算法线缓存 {idx + 1}/{total}")


def compute_depth_loss_tensor(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """返回深度 L1 损失及统计信息。"""
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"预测深度尺寸 {pred_depth.shape} 与 GT {gt_depth.shape} 不一致。"
        )
    valid_mask = gt_mask & torch.isfinite(pred_depth) & (pred_depth > 0.0)
    valid_pixels = int(valid_mask.sum().item())
    if valid_pixels == 0:
        zero = torch.zeros((), device=pred_depth.device)
        stats = {"valid_px": 0, "mae": float("nan"), "rmse": float("nan")}
        return zero, stats
    diff = pred_depth - gt_depth
    abs_diff = diff.abs()
    loss = abs_diff[valid_mask].mean()
    rmse = torch.sqrt((diff[valid_mask] ** 2).mean())
    stats = {
        "valid_px": valid_pixels,
        "mae": float(abs_diff[valid_mask].mean().detach().item()),
        "rmse": float(rmse.detach().item()),
    }
    return loss, stats


def compute_normal_loss_tensor(
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    base_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """基于余弦相似度的法线损失。"""
    if pred_normals.dim() != 3 or pred_normals.shape[0] != 3:
        raise ValueError(f"预测法线维度应为 (3,H,W)，当前为 {pred_normals.shape}")
    if gt_normals.shape != pred_normals.shape:
        raise ValueError(
            f"法线 GT 尺寸 {gt_normals.shape} 与预测 {pred_normals.shape} 不一致。"
        )
    gt_mask = torch.isfinite(gt_normals).all(dim=0)
    pred_mask = torch.isfinite(pred_normals).all(dim=0)
    valid_mask = base_mask & gt_mask & pred_mask
    valid_pixels = int(valid_mask.sum().item())
    if valid_pixels == 0:
        zero = torch.zeros((), device=pred_normals.device)
        stats = {"valid_px": 0, "mean_cos": float("nan")}
        return zero, stats

    pred_unit = F.normalize(pred_normals, dim=0)
    gt_unit = F.normalize(gt_normals, dim=0)
    cos_sim = (pred_unit * gt_unit).sum(dim=0).clamp(-1.0, 1.0)
    loss_map = (1.0 - cos_sim) * valid_mask
    loss = loss_map.sum() / valid_mask.sum()
    stats = {"valid_px": valid_pixels, "mean_cos": float(cos_sim[valid_mask].mean().item())}
    return loss, stats


class ManualScene:
    """最小 Scene 封装，提供 mesh regularization 所需的接口。"""

    def __init__(self, cameras: Sequence[Camera]):
        self._train_cameras = list(cameras)

    def getTrainCameras(self, scale: float = 1.0) -> List[Camera]:
        return list(self._train_cameras)

    def getTrainCameras_warn_up(
        self,
        iteration: int,
        warn_until_iter: int,
        scale: float = 1.0,
        scale2: float = 2.0,
    ) -> List[Camera]:
        return list(self._train_cameras)


def build_render_functions(
    gaussians: GaussianModel,
    pipe: PipelineParams,
    background: torch.Tensor,
):
    """构建训练/重建所需的渲染器：训练走 render_full 以获得精确深度梯度，SDF 仍沿用 RaDe-GS。"""

    def render_view(view: Camera) -> Dict[str, torch.Tensor]:
        pkg = render_full(
            viewpoint_camera=view,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            compute_expected_normals=False,
            compute_expected_depth=True,
            compute_accurate_median_depth_gradient=True,
        )
        if "area_max" not in pkg:
            pkg["area_max"] = torch.zeros_like(pkg["radii"])
        return pkg

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
        return {
            "render": pkg["render"].detach(),
            "median_depth": pkg["median_depth"].detach(),
        }

    return render_view, render_for_sdf


def load_mesh_config(config_name: str) -> Dict[str, Any]:
    """支持直接传文件路径或 configs/mesh/<name>.yaml。"""
    candidate = Path(config_name)
    if not candidate.is_file():
        candidate = Path(__file__).resolve().parent / "configs" / "mesh" / f"{config_name}.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(f"无法找到 mesh 配置：{config_name}")
    with candidate.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_gaussian_occupancy(gaussians: GaussianModel) -> None:
    """mesh regularization 依赖 9 维 occupancy 网格，此处在推理环境补齐缓冲。"""
    needs_init = (
        not getattr(gaussians, "learn_occupancy", False)
        or not hasattr(gaussians, "_occupancy_shift")
        or gaussians._occupancy_shift.numel() == 0
    )
    if needs_init:
        gaussians.learn_occupancy = True
        base = torch.zeros((gaussians._xyz.shape[0], 9), device=gaussians._xyz.device)
        shift = torch.zeros_like(base)
        gaussians._base_occupancy = nn.Parameter(base.requires_grad_(False), requires_grad=False)
        gaussians._occupancy_shift = nn.Parameter(shift.requires_grad_(True))


def export_mesh_from_state(
    gaussians: GaussianModel,
    mesh_state: Dict[str, Any],
    output_path: Path,
    reference_camera: Optional[Camera] = None,
) -> None:
    """根据当前 mesh_state 导出网格，并可选做视椎裁剪。"""
    gaussian_idx = mesh_state.get("delaunay_xyz_idx")
    delaunay_tets = mesh_state.get("delaunay_tets")

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

    mesh_to_save = mesh if reference_camera is None else frustum_cull_mesh(mesh, reference_camera)
    verts = mesh_to_save.verts.detach().cpu().numpy()
    faces = mesh_to_save.faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(output_path)


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """将单位四元数转换为 3x3 旋转矩阵。"""
    # 这里显式转换 DISCOVERSE 导出的四元数，确保后续符合 MILo 的旋转约定
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"四元数需要包含 4 个分量，当前形状为 {q.shape}")
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
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    return rotation


def freeze_gaussian_model(model: GaussianModel) -> None:
    """显式关闭高斯模型中参数的梯度。"""
    # 推理阶段冻结高斯参数，后续循环只做前向评估
    tensor_attrs = [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
    ]
    for attr in tensor_attrs:
        value = getattr(model, attr, None)
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)


def prepare_depth_map(depth_tensor: torch.Tensor) -> np.ndarray:
    """将深度张量转为二维 numpy 数组。"""
    # 统一 squeeze 逻辑，防止 Matplotlib 因 shape 异常报错
    depth_np = depth_tensor.detach().cpu().numpy()
    depth_np = np.squeeze(depth_np)
    if depth_np.ndim == 1:
        depth_np = np.expand_dims(depth_np, axis=0)
    return depth_np


def prepare_normals(normal_tensor: torch.Tensor) -> np.ndarray:
    """将法线张量转换为 HxWx3 的 numpy 数组。"""
    # 兼容渲染输出为 (3,H,W) 或 (H,W,3) 的两种格式
    normals_np = normal_tensor.detach().cpu().numpy()
    normals_np = np.squeeze(normals_np)
    if normals_np.ndim == 3 and normals_np.shape[0] == 3:
        normals_np = np.transpose(normals_np, (1, 2, 0))
    if normals_np.ndim == 2:
        normals_np = normals_np[..., None]
    return normals_np


def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    """将 [-1,1] 范围的法线向量映射到 [0,1] 以便可视化。"""
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = 0.5 * (normals + 1.0)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def save_normal_visualization(normal_rgb: np.ndarray, output_path: Path) -> None:
    """保存法线可视化图像。"""
    plt.imsave(output_path, normal_rgb)


def load_cameras_from_json(
    json_path: str,
    image_height: int,
    image_width: int,
    fov_y_deg: float,
) -> List[Camera]:
    """按照 MILo 的视图约定读取相机文件，并进行坐标系转换。"""
    # 自定义读取 DISCOVERSE 风格 JSON，统一转换到 COLMAP 世界->相机坐标
    pose_path = Path(json_path)
    if not pose_path.is_file():
        raise FileNotFoundError(f"未找到相机 JSON：{json_path}")

    with pose_path.open("r", encoding="utf-8") as fh:
        camera_list = json.load(fh)

    if isinstance(camera_list, dict):
        for key in ("frames", "poses", "camera_poses"):
            if key in camera_list and isinstance(camera_list[key], list):
                camera_list = camera_list[key]
                break
        else:
            raise ValueError(f"{json_path} 中的 JSON 结构不包含可识别的相机列表。")

    if not isinstance(camera_list, list) or not camera_list:
        raise ValueError(f"{json_path} 中没有有效的相机条目。")

    fov_y = math.radians(fov_y_deg)
    aspect_ratio = image_width / image_height
    fov_x = 2.0 * math.atan(aspect_ratio * math.tan(fov_y * 0.5))

    cameras: List[Camera] = []
    for idx, entry in enumerate(camera_list):
        if "quaternion" in entry:
            rotation_c2w = quaternion_to_rotation_matrix(entry["quaternion"]).astype(np.float32)
        elif "rotation" in entry:
            rotation_c2w = np.asarray(entry["rotation"], dtype=np.float32)
        else:
            raise KeyError(f"相机条目 {idx} 未提供 quaternion 或 rotation。")

        if rotation_c2w.shape != (3, 3):
            raise ValueError(f"相机条目 {idx} 的旋转矩阵形状应为 (3,3)，实际为 {rotation_c2w.shape}")

        # DISCOVER-SE 的 quaternion/rotation 直接导入后，渲染出来的 PNG 会上下翻转，
        # 说明其前进方向仍是 OpenGL 的 -Z。通过右乘 diag(1,-1,-1) 将其显式转换到
        # MILo/colmap 的坐标系，使得后续投影矩阵与深度图一致。
        rotation_c2w = rotation_c2w @ OPENGL_TO_COLMAP

        if "position" in entry:
            camera_center = np.asarray(entry["position"], dtype=np.float32)
            if camera_center.shape != (3,):
                raise ValueError(f"相机条目 {idx} 的 position 应为 3 维向量，实际为 {camera_center.shape}")
            rotation_w2c = rotation_c2w.T
            translation = (-rotation_w2c @ camera_center).astype(np.float32)
        elif "translation" in entry:
            translation = np.asarray(entry["translation"], dtype=np.float32)
            # 如果 JSON 已直接存储 colmap 风格的 T（即世界到相机），这里假设它与旋转
            # 一样来自 OpenGL 坐标。严格来说也应执行同样的轴变换，但现有数据集只有
            # position 字段；为避免重复转换，这里只做类型检查并保留原值。
        elif "tvec" in entry:
            translation = np.asarray(entry["tvec"], dtype=np.float32)
        else:
            raise KeyError(f"相机条目 {idx} 未提供 position/translation/tvec 信息。")

        if translation.shape != (3,):
            raise ValueError(f"相机条目 {idx} 的平移向量应为长度 3，实际为 {translation.shape}")

        image_name = (
            entry.get("name")
            or entry.get("image_name")
            or entry.get("img_name")
            or f"view_{idx:04d}"
        )

        camera = Camera(
            colmap_id=str(idx),
            R=rotation_c2w,
            T=translation,
            FoVx=fov_x,
            FoVy=fov_y,
            image=torch.zeros(3, image_height, image_width),
            gt_alpha_mask=None,
            image_name=image_name,
            uid=idx,
            data_device="cuda",
        )
        cameras.append(camera)
    return cameras


def save_heatmap(data: np.ndarray, output_path: Path, title: str) -> None:
    """将二维数据保存为热力图，便于直观观察差异。"""
    # 迭代间深度 / 法线差分可视化，快速定位局部变化
    plt.figure(figsize=(6, 4))
    plt.imshow(data, cmap="inferno")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = ArgumentParser(description="桥梁场景高斯到网格迭代分析脚本")
    parser.add_argument("--num_iterations", type=int, default=5, help="执行循环的次数")
    parser.add_argument("--ma_beta", type=float, default=0.8, help="loss 滑动平均系数")
    parser.add_argument(
        "--depth_loss_weight", type=float, default=0.3, help="深度一致性项权重（默认 0.3）"
    )
    parser.add_argument(
        "--normal_loss_weight", type=float, default=0.05, help="法线一致性项权重（默认 0.05）"
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        dest="lr",
        type=float,
        default=1e-3,
        help="XYZ 学习率（默认 1e-3）",
    )
    parser.add_argument(
        "--shape_lr",
        type=float,
        default=5e-4,
        help="缩放/旋转/不透明度的学习率（默认 5e-4）",
    )
    parser.add_argument(
        "--delaunay_reset_interval",
        type=int,
        default=1000,
        help="每隔多少次迭代重建一次 Delaunay（<=0 表示每次重建）",
    )
    parser.add_argument(
        "--mesh_config",
        type=str,
        default="medium",
        help="mesh 配置名称或路径（默认 medium）",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="保存可视化/npz 的间隔，默认与 Delaunay 重建间隔相同",
    )
    parser.add_argument(
        "--heatmap_dir",
        type=str,
        default="yufu2mesh_outputs",
        help="保存热力图等输出的目录",
    )
    parser.add_argument(
        "--depth_gt_dir",
        type=str,
        default="/home/zoyo/Desktop/MILo_rtx50/milo/data/bridge_clean/depth",
        help="Discoverse 深度 npy 所在目录",
    )
    parser.add_argument(
        "--depth_clip_min",
        type=float,
        default=0.0,
        help="深度最小裁剪值，<=0 表示不裁剪",
    )
    parser.add_argument(
        "--depth_clip_max",
        type=float,
        default=None,
        help="深度最大裁剪值，None 表示不裁剪",
    )
    parser.add_argument(
        "--normal_cache_dir",
        type=str,
        default=None,
        help="法线缓存目录，默认为 runs/<heatmap_dir>/normal_gt",
    )
    parser.add_argument(
        "--skip_normal_gt_generation",
        action="store_true",
        help="已存在缓存时跳过初始法线 GT 预计算",
    )
    parser.add_argument("--seed", type=int, default=0, help="控制随机性的种子")
    parser.add_argument(
        "--lock_view_index",
        type=int,
        default=None,
        help="固定视角索引，仅在指定时输出热力图",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="控制迭代日志的打印频率（默认每次迭代打印）",
    )
    parser.add_argument(
        "--warn_until_iter",
        type=int,
        default=3000,
        help="surface sampling warmup 迭代数（用于 mesh downsample）",
    )
    parser.add_argument(
        "--imp_metric",
        type=str,
        default="outdoor",
        choices=["outdoor", "indoor"],
        help="surface sampling 的重要性度量类型",
    )
    parser.add_argument(
        "--mesh_start_iter",
        type=int,
        default=2000,
        help="mesh 正则起始迭代（默认 2000，避免冷启动阶段干扰）",
    )
    parser.add_argument(
        "--mesh_update_interval",
        type=int,
        default=5,
        help="mesh 正则重建/回传间隔，>1 可减少 DMTet 抖动（默认 5）",
    )
    parser.add_argument(
        "--mesh_depth_weight",
        type=float,
        default=0.1,
        help="mesh 深度项权重覆盖（默认 0.1，原配置通常为 0.05）",
    )
    parser.add_argument(
        "--mesh_normal_weight",
        type=float,
        default=0.1,
        help="mesh 法线项权重覆盖（默认 0.1，原配置通常为 0.05）",
    )
    parser.add_argument(
        "--disable_shape_training",
        action="store_true",
        help="禁用缩放/旋转/不透明度的优化，仅用于调试",
    )
    pipe = PipelineParams(parser)
    args = parser.parse_args()

    pipe.debug = getattr(args, "debug", False)

    # 所有输出固定写入 milo/runs/ 下，便于管理实验产物
    base_run_dir = Path(__file__).resolve().parent / "runs"
    output_dir = base_run_dir / args.heatmap_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ply_path = "/home/zoyo/Desktop/MILo_rtx50/milo/data/bridge_clean/yufu_bridge_cleaned.ply"
    camera_poses_json = "/home/zoyo/Desktop/MILo_rtx50/milo/data/bridge_clean/camera_poses_cam1.json"

    gaussians = GaussianModel(sh_degree=0, learn_occupancy=True)
    gaussians.load_ply(ply_path)
    freeze_attrs = [
        "_features_dc",
        "_features_rest",
        "_base_occupancy",
        "_occupancy_shift",
    ]
    for attr in freeze_attrs:
        value = getattr(gaussians, attr, None)
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)
    gaussians._xyz.requires_grad_(True)

    shape_trainable = []
    shape_attr_list = ["_scaling", "_rotation", "_opacity"]
    if not args.disable_shape_training:
        for attr in shape_attr_list:
            value = getattr(gaussians, attr, None)
            if isinstance(value, torch.Tensor):
                value.requires_grad_(True)
                shape_trainable.append(value)
    else:
        for attr in shape_attr_list:
            value = getattr(gaussians, attr, None)
            if isinstance(value, torch.Tensor):
                value.requires_grad_(False)

    height = 720
    width = 1280
    fov_y_deg = 75.0

    train_cameras = load_cameras_from_json(
        json_path=camera_poses_json,
        image_height=height,
        image_width=width,
        fov_y_deg=fov_y_deg,
    )
    num_views = len(train_cameras)
    print(f"[INFO] 成功加载 {num_views} 个相机视角。")

    device = gaussians._xyz.device
    background = torch.tensor([0.0, 0.0, 0.0], device=device)

    mesh_config = load_mesh_config(args.mesh_config)
    mesh_config["start_iter"] = max(0, args.mesh_start_iter)
    mesh_config["stop_iter"] = max(mesh_config.get("stop_iter", args.num_iterations), args.num_iterations)
    mesh_config["mesh_update_interval"] = max(1, args.mesh_update_interval)
    mesh_config["delaunay_reset_interval"] = args.delaunay_reset_interval
    mesh_config["depth_weight"] = args.mesh_depth_weight
    mesh_config["normal_weight"] = args.mesh_normal_weight
    # 这里默认沿用 surface 采样以对齐训练阶段；如仅需快速分析，也可以切换为 random 提升速度。
    mesh_config["delaunay_sampling_method"] = "surface"

    scene_wrapper = ManualScene(train_cameras)

    ensure_gaussian_occupancy(gaussians)
    if gaussians.spatial_lr_scale <= 0:
        gaussians.spatial_lr_scale = 1.0
    gaussians.set_occupancy_mode(mesh_config.get("occupancy_mode", "occupancy_shift"))

    render_view, render_for_sdf = build_render_functions(gaussians, pipe, background)

    depth_clip_min = args.depth_clip_min if args.depth_clip_min > 0.0 else None
    depth_clip_max = args.depth_clip_max
    depth_provider = DepthProvider(
        depth_root=Path(args.depth_gt_dir),
        image_height=height,
        image_width=width,
        device=device,
        clip_min=depth_clip_min,
        clip_max=depth_clip_max,
    )

    normal_cache_dir = Path(args.normal_cache_dir) if args.normal_cache_dir else (output_dir / "normal_gt")
    normal_cache = NormalGroundTruthCache(
        cache_dir=normal_cache_dir,
        image_height=height,
        image_width=width,
        device=device,
    )
    if args.skip_normal_gt_generation:
        missing = [idx for idx in range(num_views) if not normal_cache.has(idx)]
        if missing:
            raise RuntimeError(
                f"跳过法线 GT 预计算被拒绝，仍有 {len(missing)} 个视角缺少缓存（示例 {missing[:5]}）。"
            )
    else:
        print("[INFO] 开始预计算初始法线 GT（仅进行一次，若存在缓存会自动跳过）。")
        normal_cache.ensure_all(train_cameras, render_view)
        normal_cache.clear_memory_cache()

    mesh_renderer, mesh_state = initialize_mesh_regularization(scene_wrapper, mesh_config)
    mesh_state["reset_delaunay_samples"] = True
    mesh_state["reset_sdf_values"] = True

    param_groups = [{"params": [gaussians._xyz], "lr": args.lr}]
    if shape_trainable:
        param_groups.append({"params": shape_trainable, "lr": args.shape_lr})
    optimizer = torch.optim.Adam(param_groups)
    mesh_args = SimpleNamespace(
        warn_until_iter=args.warn_until_iter,
        imp_metric=args.imp_metric,
    )

    # 记录整个迭代过程中的指标与梯度，结束时统一写入 npz/曲线
    stats_history: Dict[str, List[float]] = {
        "iteration": [],
        "depth_loss": [],
        "normal_loss": [],
        "mesh_loss": [],
        "mesh_depth_loss": [],
        "mesh_normal_loss": [],
        "occupied_centers_loss": [],
        "occupancy_labels_loss": [],
        "depth_mae": [],
        "depth_rmse": [],
        "normal_mean_cos": [],
        "normal_valid_px": [],
        "grad_norm": [],
    }

    moving_loss = None
    previous_depth: Dict[int, np.ndarray] = {}
    previous_normals: Dict[int, np.ndarray] = {}
    camera_stack = list(range(num_views))
    random.shuffle(camera_stack)
    save_interval = args.save_interval if args.save_interval is not None else args.delaunay_reset_interval
    if save_interval is None or save_interval <= 0:
        save_interval = 1

    for iteration in range(args.num_iterations):
        optimizer.zero_grad(set_to_none=True)
        if args.lock_view_index is not None:
            view_index = args.lock_view_index % num_views
        else:
            if not camera_stack:
                camera_stack = list(range(num_views))
                random.shuffle(camera_stack)
            view_index = camera_stack.pop()
        viewpoint = train_cameras[view_index]

        training_pkg = render_view(viewpoint)
        gt_depth_tensor, gt_depth_mask = depth_provider.get(view_index)
        depth_loss_tensor, depth_stats = compute_depth_loss_tensor(
            pred_depth=training_pkg["median_depth"],
            gt_depth=gt_depth_tensor,
            gt_mask=gt_depth_mask,
        )
        gt_normals_tensor = normal_cache.get(view_index)
        normal_loss_tensor, normal_stats = compute_normal_loss_tensor(
            pred_normals=training_pkg["normal"],
            gt_normals=gt_normals_tensor,
            base_mask=gt_depth_mask,
        )
        def _zero_mesh_pkg() -> Dict[str, Any]:
            zero = torch.zeros((), device=device)
            depth_zero = torch.zeros_like(training_pkg["median_depth"])
            normal_zero = torch.zeros_like(training_pkg["normal"].permute(1, 2, 0))
            return {
                "mesh_loss": zero,
                "mesh_depth_loss": zero,
                "mesh_normal_loss": zero,
                "occupied_centers_loss": zero,
                "occupancy_labels_loss": zero,
                "updated_state": mesh_state,
                "mesh_render_pkg": {
                    "depth": depth_zero,
                    "normals": normal_zero,
                },
            }

        mesh_active = iteration >= mesh_config["start_iter"]
        if mesh_active:
            mesh_pkg = compute_mesh_regularization(
                iteration=iteration,
                render_pkg=training_pkg,
                viewpoint_cam=viewpoint,
                viewpoint_idx=view_index,
                gaussians=gaussians,
                scene=scene_wrapper,
                pipe=pipe,
                background=background,
                kernel_size=0.0,
                config=mesh_config,
                mesh_renderer=mesh_renderer,
                mesh_state=mesh_state,
                render_func=render_for_sdf,
                weight_adjustment=1.0,
                args=mesh_args,
                integrate_func=integrate_radegs,
            )
        else:
            mesh_pkg = _zero_mesh_pkg()

        mesh_state = mesh_pkg["updated_state"]
        mesh_loss_tensor = mesh_pkg["mesh_loss"]
        total_loss = (
            args.depth_loss_weight * depth_loss_tensor
            + args.normal_loss_weight * normal_loss_tensor
            + mesh_loss_tensor
        )
        depth_loss_value = float(depth_loss_tensor.detach().item())
        normal_loss_value = float(normal_loss_tensor.detach().item())
        mesh_loss_value = float(mesh_loss_tensor.detach().item())
        loss_value = float(total_loss.detach().item())

        if total_loss.requires_grad:
            total_loss.backward()
            grad_norm = float(gaussians._xyz.grad.detach().norm().item())
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            grad_norm = float("nan")

        mesh_render_pkg = mesh_pkg["mesh_render_pkg"]
        mesh_depth_map = prepare_depth_map(mesh_render_pkg["depth"])
        mesh_normals_map = prepare_normals(mesh_render_pkg["normals"])
        gaussian_depth_map = prepare_depth_map(training_pkg["median_depth"])
        gaussian_normals_map = prepare_normals(training_pkg["normal"])
        gt_depth_map = depth_provider.as_numpy(view_index)
        gt_normals_map = prepare_normals(gt_normals_tensor)

        mesh_valid = np.isfinite(mesh_depth_map) & (mesh_depth_map > 0.0)
        gaussian_valid = np.isfinite(gaussian_depth_map) & (gaussian_depth_map > 0.0)
        gt_valid = np.isfinite(gt_depth_map) & (gt_depth_map > 0.0)
        overlap_mask = gaussian_valid & gt_valid

        depth_delta = gaussian_depth_map - gt_depth_map
        if overlap_mask.any():
            delta_abs = np.abs(depth_delta[overlap_mask])
            diff_mean = float(delta_abs.mean())
            diff_max = float(delta_abs.max())
            diff_rmse = float(np.sqrt(np.mean(depth_delta[overlap_mask] ** 2)))
        else:
            diff_mean = diff_max = diff_rmse = float("nan")

        mesh_depth_loss = float(mesh_pkg["mesh_depth_loss"].item())
        mesh_normal_loss = float(mesh_pkg["mesh_normal_loss"].item())
        occupied_loss = float(mesh_pkg["occupied_centers_loss"].item())
        labels_loss = float(mesh_pkg["occupancy_labels_loss"].item())

        moving_loss = (
            loss_value
            if moving_loss is None
            else args.ma_beta * moving_loss + (1 - args.ma_beta) * loss_value
        )

        stats_history["iteration"].append(float(iteration))
        stats_history["depth_loss"].append(depth_loss_value)
        stats_history["normal_loss"].append(normal_loss_value)
        stats_history["mesh_loss"].append(mesh_loss_value)
        stats_history["mesh_depth_loss"].append(mesh_depth_loss)
        stats_history["mesh_normal_loss"].append(mesh_normal_loss)
        stats_history["occupied_centers_loss"].append(occupied_loss)
        stats_history["occupancy_labels_loss"].append(labels_loss)
        stats_history["depth_mae"].append(depth_stats["mae"])
        stats_history["depth_rmse"].append(depth_stats["rmse"])
        stats_history["normal_mean_cos"].append(normal_stats["mean_cos"])
        stats_history["normal_valid_px"].append(float(normal_stats["valid_px"]))
        stats_history["grad_norm"].append(grad_norm)

        def _fmt(value: float) -> str:
            return f"{value:.6f}"

        if (iteration % max(1, args.log_interval) == 0) or (iteration == args.num_iterations - 1):
            print(
                "[INFO] Iter {iter:02d} | loss={total} (depth={depth}, normal={normal}, mesh={mesh}) | ma_loss={ma}".format(
                    iter=iteration,
                    total=_fmt(loss_value),
                    depth=_fmt(depth_loss_value),
                    normal=_fmt(normal_loss_value),
                    mesh=_fmt(mesh_loss_value),
                    ma=f"{moving_loss:.6f}",
                )
            )

        should_save = (save_interval <= 0) or (iteration % save_interval == 0)
        if should_save:
            valid_values: List[np.ndarray] = []
            if mesh_valid.any():
                valid_values.append(mesh_depth_map[mesh_valid].reshape(-1))
            if gaussian_valid.any():
                valid_values.append(gaussian_depth_map[gaussian_valid].reshape(-1))
            if gt_valid.any():
                valid_values.append(gt_depth_map[gt_valid].reshape(-1))
            if valid_values:
                all_valid = np.concatenate(valid_values)
                shared_min = float(all_valid.min())
                shared_max = float(all_valid.max())
            else:
                shared_min, shared_max = 0.0, 1.0

            gaussian_depth_vis_path = output_dir / f"gaussian_depth_vis_iter_{iteration:02d}.png"
            plt.imsave(
                gaussian_depth_vis_path,
                gaussian_depth_map,
                cmap="viridis",
                vmin=shared_min,
                vmax=shared_max,
            )

            depth_vis_path = output_dir / f"mesh_depth_vis_iter_{iteration:02d}.png"
            plt.imsave(
                depth_vis_path,
                mesh_depth_map,
                cmap="viridis",
                vmin=shared_min,
                vmax=shared_max,
            )

            gt_depth_vis_path = output_dir / f"gt_depth_vis_iter_{iteration:02d}.png"
            plt.imsave(
                gt_depth_vis_path,
                gt_depth_map,
                cmap="viridis",
                vmin=shared_min,
                vmax=shared_max,
            )

            normal_vis_path = output_dir / f"mesh_normal_vis_iter_{iteration:02d}.png"
            mesh_normals_rgb = normals_to_rgb(mesh_normals_map)
            save_normal_visualization(mesh_normals_rgb, normal_vis_path)

            gaussian_normal_vis_path = output_dir / f"gaussian_normal_vis_iter_{iteration:02d}.png"
            gaussian_normals_rgb = normals_to_rgb(gaussian_normals_map)
            save_normal_visualization(gaussian_normals_rgb, gaussian_normal_vis_path)

            gt_normal_vis_path = output_dir / f"gt_normal_vis_iter_{iteration:02d}.png"
            gt_normals_rgb = normals_to_rgb(gt_normals_map)
            save_normal_visualization(gt_normals_rgb, gt_normal_vis_path)

            output_npz = output_dir / f"mesh_render_iter_{iteration:02d}.npz"
            np.savez(
                output_npz,
                mesh_depth=mesh_depth_map,
                gaussian_depth=gaussian_depth_map,
                depth_gt=gt_depth_map,
                mesh_normals=mesh_normals_map,
                gaussian_normals=gaussian_normals_map,
                normal_gt=gt_normals_map,
                depth_loss=depth_loss_value,
                normal_loss=normal_loss_value,
                mesh_loss=mesh_loss_value,
                mesh_depth_loss=mesh_depth_loss,
                mesh_normal_loss=mesh_normal_loss,
                occupied_centers_loss=occupied_loss,
                occupancy_labels_loss=labels_loss,
                loss=loss_value,
                moving_loss=moving_loss,
                depth_mae=depth_stats["mae"],
                depth_rmse=depth_stats["rmse"],
                normal_valid_px=normal_stats["valid_px"],
                normal_mean_cos=normal_stats["mean_cos"],
                grad_norm=grad_norm,
                iteration=iteration,
                learning_rate=optimizer.param_groups[0]["lr"],
            )

            if overlap_mask.any():
                depth_diff_vis = np.zeros_like(gaussian_depth_map)
                depth_diff_vis[overlap_mask] = depth_delta[overlap_mask]
                save_heatmap(
                    np.abs(depth_diff_vis),
                    output_dir / f"depth_diff_iter_{iteration:02d}.png",
                    f"|Pred-GT| iter {iteration}",
                )

            composite_path = output_dir / f"comparison_iter_{iteration:02d}.png"
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
            ax_gt_depth, ax_gaussian_depth, ax_mesh_depth = axes[0]
            ax_gt_normals, ax_gaussian_normals, ax_mesh_normals = axes[1]

            im0 = ax_gt_depth.imshow(gt_depth_map, cmap="viridis", vmin=shared_min, vmax=shared_max)
            ax_gt_depth.set_title("GT depth")
            ax_gt_depth.axis("off")
            fig.colorbar(im0, ax=ax_gt_depth, fraction=0.046, pad=0.04)

            im1 = ax_gaussian_depth.imshow(gaussian_depth_map, cmap="viridis", vmin=shared_min, vmax=shared_max)
            ax_gaussian_depth.set_title("Gaussian depth")
            ax_gaussian_depth.axis("off")
            fig.colorbar(im1, ax=ax_gaussian_depth, fraction=0.046, pad=0.04)

            im2 = ax_mesh_depth.imshow(mesh_depth_map, cmap="viridis", vmin=shared_min, vmax=shared_max)
            ax_mesh_depth.set_title("Mesh depth")
            ax_mesh_depth.axis("off")
            fig.colorbar(im2, ax=ax_mesh_depth, fraction=0.046, pad=0.04)

            ax_gt_normals.imshow(gt_normals_rgb)
            ax_gt_normals.set_title("GT normals")
            ax_gt_normals.axis("off")

            ax_gaussian_normals.imshow(gaussian_normals_rgb)
            ax_gaussian_normals.set_title("Gaussian normals")
            ax_gaussian_normals.axis("off")

            ax_mesh_normals.imshow(mesh_normals_rgb)
            ax_mesh_normals.set_title("Mesh normals")
            ax_mesh_normals.axis("off")

            info_lines = [
                f"Iteration: {iteration:02d}",
                f"View index: {view_index}",
                f"GT depth valid px: {int(gt_valid.sum())}",
                f"Gaussian depth valid px: {int(gaussian_valid.sum())}",
                f"|Pred - GT| mean={diff_mean:.3f}, max={diff_max:.3f}, RMSE={diff_rmse:.3f}",
                f"Depth loss={_fmt(depth_loss_value)} (w={args.depth_loss_weight:.2f}, mae={depth_stats['mae']:.3f}, rmse={depth_stats['rmse']:.3f})",
                f"Normal loss={_fmt(normal_loss_value)} (w={args.normal_loss_weight:.2f}, px={normal_stats['valid_px']}, cos={normal_stats['mean_cos']:.3f})",
                f"Mesh loss={_fmt(mesh_loss_value)}",
                f"Mesh depth loss={_fmt(mesh_depth_loss)} mesh normal loss={_fmt(mesh_normal_loss)}",
                f"Occupied centers={_fmt(occupied_loss)} labels={_fmt(labels_loss)}",
            ]
            fig.suptitle("\n".join(info_lines), fontsize=12, y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(composite_path, dpi=300)
            plt.close(fig)

            if args.lock_view_index is not None:
                if view_index in previous_depth:
                    depth_diff = np.abs(gaussian_depth_map - previous_depth[view_index])
                    save_heatmap(
                        depth_diff,
                        output_dir / f"depth_diff_iter_{iteration:02d}_temporal.png",
                        f"Depth Δ iter {iteration}",
                    )
                if view_index in previous_normals:
                    normal_delta = gaussian_normals_map - previous_normals[view_index]
                    if normal_delta.ndim == 3:
                        normal_diff = np.linalg.norm(normal_delta, axis=-1)
                    else:
                        normal_diff = np.abs(normal_delta)
                    save_heatmap(
                        normal_diff,
                        output_dir / f"normal_diff_iter_{iteration:02d}_temporal.png",
                        f"Normal Δ iter {iteration}",
                    )

            with torch.no_grad():
                export_mesh_from_state(
                    gaussians=gaussians,
                    mesh_state=mesh_state,
                    output_path=output_dir / f"mesh_iter_{iteration:02d}.ply",
                    reference_camera=None,
                )

        if args.lock_view_index is not None:
            previous_depth[view_index] = gaussian_depth_map
            previous_normals[view_index] = gaussian_normals_map
    with torch.no_grad():
        # 输出完整指标轨迹及汇总曲线，方便任务结束后快速复盘
        history_npz = output_dir / "metrics_history.npz"
        np.savez(
            history_npz,
            **{k: np.asarray(v, dtype=np.float32) for k, v in stats_history.items()},
        )
        summary_fig = output_dir / "metrics_summary.png"
        if stats_history["iteration"]:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=200)
            iters = np.asarray(stats_history["iteration"])
            axes[0, 0].plot(iters, stats_history["depth_loss"], label="depth")
            axes[0, 0].plot(iters, stats_history["normal_loss"], label="normal")
            axes[0, 0].plot(iters, stats_history["mesh_loss"], label="mesh")
            axes[0, 0].set_title("Total losses")
            axes[0, 0].set_xlabel("Iteration")
            axes[0, 0].legend()

            axes[0, 1].plot(iters, stats_history["mesh_depth_loss"], label="mesh depth")
            axes[0, 1].plot(iters, stats_history["mesh_normal_loss"], label="mesh normal")
            axes[0, 1].plot(iters, stats_history["occupied_centers_loss"], label="occupied centers")
            axes[0, 1].plot(iters, stats_history["occupancy_labels_loss"], label="occupancy labels")
            axes[0, 1].set_title("Mesh regularization components")
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].legend()

            axes[1, 0].plot(iters, stats_history["depth_mae"], label="depth MAE")
            axes[1, 0].plot(iters, stats_history["depth_rmse"], label="depth RMSE")
            axes[1, 0].set_title("Depth metrics")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].legend()

            axes[1, 1].plot(iters, stats_history["normal_mean_cos"], label="mean cos")
            axes[1, 1].plot(iters, stats_history["grad_norm"], label="grad norm")
            axes[1, 1].set_title("Normals / Gradients")
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].legend()

            fig.tight_layout()
            fig.savefig(summary_fig)
            plt.close(fig)
            print(f"[INFO] 已保存曲线汇总：{summary_fig}")
        print(f"[INFO] 记录所有迭代指标到 {history_npz}")
        final_mesh_path = output_dir / "mesh_final.ply"
        final_gaussian_path = output_dir / "gaussians_final.ply"
        print(f"[INFO] 导出最终 mesh 到 {final_mesh_path}")
        export_mesh_from_state(
            gaussians=gaussians,
            mesh_state=mesh_state,
            output_path=final_mesh_path,
            reference_camera=None,
        )
        print(f"[INFO] 导出最终高斯到 {final_gaussian_path}")
        gaussians.save_ply(str(final_gaussian_path))
    print("[INFO] 循环结束，所有结果已写入输出目录。")


if __name__ == "__main__":
    main()
