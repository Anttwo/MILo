import os
import json
import math
import random
from argparse import ArgumentParser
from typing import List, Optional, Sequence

import yaml
from types import SimpleNamespace
import torch
import torch.nn as nn
import numpy as np
import trimesh

from arguments import PipelineParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from regularization.regularizer.mesh import initialize_mesh_regularization, compute_mesh_regularization
from functional import extract_mesh, compute_delaunay_triangulation
from functional.mesh import frustum_cull_mesh
from regularization.sdf.learnable import convert_occupancy_to_sdf
from utils.geometry_utils import flatten_voronoi_features
from gaussian_renderer.radegs import render_radegs, integrate_radegs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def quaternion_to_rotation_matrix(q: Sequence[float]) -> np.ndarray:
    """Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix."""
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


class ManualScene:
    """Minimal scene wrapper exposing the API expected by mesh regularization utilities."""

    def __init__(self, cameras: Sequence[Camera]):
        self._train_cameras = list(cameras)

    def getTrainCameras(self, scale: float = 1.0):
        return list(self._train_cameras)

    def getTrainCameras_warn_up(
        self,
        iteration: int,
        warn_until_iter: int,
        scale: float = 1.0,
        scale2: float = 2.0,
    ):
        return list(self._train_cameras)


def load_cameras_from_json(
    json_path: str,
    image_height: int,
    image_width: int,
    fov_y_deg: float,
) -> List[Camera]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Camera JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        camera_entries = json.load(f)

    if not camera_entries:
        raise ValueError(f"No camera entries found in {json_path}")

    fov_y = math.radians(fov_y_deg)
    aspect_ratio = image_width / image_height
    fov_x = 2.0 * math.atan(aspect_ratio * math.tan(fov_y * 0.5))

    cameras: List[Camera] = []
    for idx, entry in enumerate(camera_entries):
        if "quaternion" in entry:
            rotation = quaternion_to_rotation_matrix(entry["quaternion"])
        elif "rotation" in entry:
            rotation = np.asarray(entry["rotation"], dtype=np.float32)
            if rotation.shape != (3, 3):
                raise ValueError(f"Camera entry {idx} rotation must be 3x3, got {rotation.shape}")
        else:
            raise KeyError(f"Camera entry {idx} must provide either 'quaternion' or 'rotation'.")

        translation = None
        if "tvec" in entry:
            translation = np.asarray(entry["tvec"], dtype=np.float32)
        elif "translation" in entry:
            translation = np.asarray(entry["translation"], dtype=np.float32)
        elif "position" in entry:
            camera_center = np.asarray(entry["position"], dtype=np.float32)
            if camera_center.shape != (3,):
                raise ValueError(f"Camera entry {idx} position must be length-3, got shape {camera_center.shape}")
            # Camera expects world-to-view translation (COLMAP convention t = -R * C).
            rotation_w2c = rotation.T  # rotation is camera-to-world
            translation = -rotation_w2c @ camera_center
        else:
            raise KeyError(f"Camera entry {idx} must provide 'position', 'translation', or 'tvec'.")

        if translation.shape != (3,):
            raise ValueError(f"Camera entry {idx} translation must be length-3, got shape {translation.shape}")

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
            data_device="cuda",
        )
        cameras.append(camera)
    return cameras


def freeze_gaussian_rigid_parameters(gaussians: GaussianModel) -> None:
    """Disable gradients for geometric and appearance parameters, keeping occupancy shift trainable."""
    freeze_attrs = [
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
    ]
    for attr in freeze_attrs:
        param = getattr(gaussians, attr, None)
        if isinstance(param, nn.Parameter):
            param.requires_grad_(False)

    if hasattr(gaussians, "_base_occupancy") and isinstance(gaussians._base_occupancy, nn.Parameter):
        gaussians._base_occupancy.requires_grad_(False)
    if hasattr(gaussians, "_occupancy_shift") and isinstance(gaussians._occupancy_shift, nn.Parameter):
        gaussians._occupancy_shift.requires_grad_(True)


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


def load_mesh_config_file(name: str) -> dict:
    config_path = os.path.join(BASE_DIR, "configs", "mesh", f"{name}.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Mesh config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_mesh_from_gaussians(
    gaussians: GaussianModel,
    mesh_state: dict,
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

    mesh_to_save = mesh
    if reference_camera is not None:
        mesh_to_save = frustum_cull_mesh(mesh, reference_camera)

    verts = mesh_to_save.verts.detach().cpu().numpy()
    faces = mesh_to_save.faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(output_path)


def train_occupancy_only(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for occupancy fine-tuning.")

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

    scene = ManualScene(cameras)

    mesh_config = load_mesh_config_file(args.mesh_config)
    mesh_config["start_iter"] = max(1, args.mesh_start_iter)
    if args.mesh_stop_iter is not None:
        mesh_config["stop_iter"] = args.mesh_stop_iter
    else:
        mesh_config["stop_iter"] = max(mesh_config.get("stop_iter", args.iterations), args.iterations)

    pipe_parser = ArgumentParser()
    pipe = PipelineParams(pipe_parser)

    background = torch.tensor(args.background, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(
        sh_degree=0,
        use_mip_filter=False,
        learn_occupancy=True,
        use_appearance_network=False,
    )
    gaussians.load_ply(args.ply_path)
    print(f"[INFO] Loaded {gaussians._xyz.shape[0]} Gaussians from {args.ply_path}.")

    if not gaussians.learn_occupancy or not hasattr(gaussians, "_occupancy_shift"):
        print("[INFO] PLY does not provide occupancy buffers; initializing them to zeros.")
        gaussians.learn_occupancy = True
        base_occupancy = torch.zeros((gaussians._xyz.shape[0], 9), device=gaussians._xyz.device)
        occupancy_shift = torch.zeros_like(base_occupancy)
        gaussians._base_occupancy = nn.Parameter(base_occupancy.requires_grad_(False), requires_grad=False)
        gaussians._occupancy_shift = nn.Parameter(occupancy_shift.requires_grad_(True))

    if gaussians.spatial_lr_scale <= 0:
        gaussians.spatial_lr_scale = 1.0

    gaussians.init_culling(len(cameras))
    gaussians.set_occupancy_mode(mesh_config.get("occupancy_mode", "occupancy_shift"))
    freeze_gaussian_rigid_parameters(gaussians)

    optimizer = torch.optim.Adam([gaussians._occupancy_shift], lr=args.occupancy_lr)

    render_view, render_for_sdf = build_render_functions(gaussians, pipe, background)
    mesh_renderer, mesh_state = initialize_mesh_regularization(scene, mesh_config)
    mesh_state["reset_delaunay_samples"] = True
    mesh_state["reset_sdf_values"] = True

    runtime_args = SimpleNamespace(
        warn_until_iter=args.warn_until_iter,
        imp_metric=args.imp_metric,
        depth_reinit_iter=getattr(args, 'depth_reinit_iter', args.warn_until_iter),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    ema_loss: Optional[float] = None

    for iteration in range(1, args.iterations + 1):
        view_idx = random.randrange(len(cameras))
        viewpoint = cameras[view_idx]

        render_pkg = render_view(viewpoint)

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

        if mesh_pkg.get("mesh_triangles") is not None and mesh_pkg["mesh_triangles"].numel() == 0:
            print(f"[WARNING] Empty mesh at iteration {iteration}; skipping optimizer step.")
            continue

        loss = mesh_pkg["mesh_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        ema_loss = loss_value if ema_loss is None else (0.9 * ema_loss + 0.1 * loss_value)

        if iteration % args.log_interval == 0 or iteration == 1:
            print(
                "[Iter {iter:05d}] loss={loss:.6f} ema={ema:.6f} depth={depth:.6f} "
                "normal={normal:.6f} occ_centers={centers:.6f} labels={labels:.6f}".format(
                    iter=iteration,
                    loss=loss_value,
                    ema=ema_loss,
                    depth=mesh_pkg["mesh_depth_loss"].item(),
                    normal=mesh_pkg["mesh_normal_loss"].item(),
                    centers=mesh_pkg["occupied_centers_loss"].item(),
                    labels=mesh_pkg["occupancy_labels_loss"].item(),
                )
            )

        if args.export_interval > 0 and iteration % args.export_interval == 0:
            iteration_dir = os.path.join(args.output_dir, f"iter_{iteration:05d}")
            os.makedirs(iteration_dir, exist_ok=True)
            gaussians.save_ply(os.path.join(iteration_dir, "point_cloud.ply"))
            export_mesh_from_gaussians(
                gaussians=gaussians,
                mesh_state=mesh_state,
                output_path=os.path.join(iteration_dir, "mesh.ply"),
                reference_camera=cameras[0] if args.cull_on_export else None,
            )

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(final_dir, "point_cloud.ply"))
    export_mesh_from_gaussians(
        gaussians=gaussians,
        mesh_state=mesh_state,
        output_path=os.path.join(final_dir, "mesh.ply"),
        reference_camera=cameras[0] if args.cull_on_export else None,
    )

    print(f"[INFO] Occupancy-only training completed. Results saved to {args.output_dir}.")


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Occupancy-only fine-tuning from a pretrained Gaussian PLY.")
    parser.add_argument("--ply_path", type=str, required=True, help="Input PLY file with pretrained Gaussians.")
    parser.add_argument("--camera_poses", type=str, required=True, help="JSON file containing camera poses.")
    parser.add_argument("--mesh_config", type=str, default="default", help="Mesh regularization config name.")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of optimization steps.")
    parser.add_argument("--occupancy_lr", type=float, default=0.01, help="Learning rate for occupancy shift.")
    parser.add_argument("--log_interval", type=int, default=50, help="Console logging interval.")
    parser.add_argument("--export_interval", type=int, default=200, help="Mesh/PLY export interval.")
    parser.add_argument("--output_dir", type=str, default="./ply2mesh_output", help="Directory to store outputs.")
    parser.add_argument("--fov_y", type=float, default=75.0, help="Vertical field-of-view in degrees.")
    parser.add_argument("--image_width", type=int, default=1280, help="Rendered image width.")
    parser.add_argument("--image_height", type=int, default=720, help="Rendered image height.")
    parser.add_argument(
        "--background",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Background color used for rendering (RGB).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--mesh_start_iter", type=int, default=1, help="Iteration at which mesh regularization starts.")
    parser.add_argument(
        "--mesh_stop_iter",
        type=int,
        default=None,
        help="Iteration after which mesh regularization stops (defaults to total iterations).",
    )
    parser.add_argument("--warn_until_iter", type=int, default=3000, help="Warmup iterations for surface sampling.")
    parser.add_argument("--imp_metric", type=str, default="outdoor", choices=["outdoor", "indoor"], help="Importance metric for surface sampling.")
    parser.add_argument("--cull_on_export", action="store_true", help="Enable frustum culling using the first camera before exporting meshes.")
    return parser


if __name__ == "__main__":
    argument_parser = build_arg_parser()
    parsed_args = argument_parser.parse_args()
    train_occupancy_only(parsed_args)
