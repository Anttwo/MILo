from functional import (
    sample_gaussians_on_surface,
    extract_gaussian_pivots,
    compute_initial_sdf_values,
    compute_delaunay_triangulation,
    extract_mesh,
    frustum_cull_mesh,
)
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
import json, math, torch, trimesh
import numpy as np
from arguments import ModelParams, PipelineParams, OptimizationParams, read_config
def quaternion_to_rotation_matrix(q):
    """
    将单位四元数转换为3x3旋转矩阵。

    参数:
        q: 一个包含四个元素的列表或数组 [w, x, y, z]

    返回:
        R: 一个3x3的NumPy数组表示的旋转矩阵。
    """
    w, x, y, z = q
    # 计算矩阵的每个元素，避免重复计算以提高效率
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    R = np.array([
        [1 - 2 * (yy + zz),      2 * (xy - wz),       2 * (xz + wy)],
        [2 * (xy + wz),        1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),          2 * (yz + wx),     1 - 2 * (xx + yy)]
    ])
    return R

# Load or initialize a 3DGS-like model and training cameras
ply_path = "/home/zoyo/Desktop/MILo_rtx50/milo/data/Bridge/yufu_bridge_cleaned.ply"
camera_poses_json = "/home/zoyo/Desktop/MILo_rtx50/milo/data/Bridge/camera_poses_cam1.json"
camera_poses = json.load(open(camera_poses_json))
with open(camera_poses_json, 'r') as fcc_file:
    fcc_data = json.load(fcc_file)
    print(len(fcc_data),type(fcc_data))

gaussians = GaussianModel(
        sh_degree=0, 
        # use_mip_filter=use_mip_filter, 
        # learn_occupancy=args.mesh_regularization,
        # use_appearance_network=args.decoupled_appearance,
    )
gaussians.load_ply(ply_path)
train_cameras = []
height = 720
width = 1280
fov_y = math.radians(75) 
# fov_x = math.radians(108)
aspect_ratio = width / height
fov_x = 2 * math.atan(aspect_ratio * math.tan(fov_y / 2))
for i in range(len(fcc_data)):
    camera_info = fcc_data[i]
    camera = Camera(
            colmap_id=str(i), 
            R=quaternion_to_rotation_matrix(camera_info['quaternion']),
            T=np.asarray(camera_info['position']), 
            FoVx=fov_x, 
            FoVy=fov_y, 
            image=torch.empty(3, height, width),
            gt_alpha_mask=None,
            image_name=camera_info['name'], 
            uid=i,
            data_device='cuda',
        )
    train_cameras.append(camera)

# following this template. It will be used only for initializing SDF values.
# The wrapper should accept just a camera as input, and return a dictionary 
# with "render" and "depth" keys.
from gaussian_renderer.radegs import render_radegs


from argparse import ArgumentParser, Namespace
parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--bug", type=bool, default=False)
pipe = PipelineParams(parser)
background = torch.tensor([0., 0., 0.], device="cuda")
def render_func(view):
    render_pkg = render_radegs(
        viewpoint_camera=view, 
        pc=gaussians, 
        pipe=pipe, 
        bg_color=background, 
        kernel_size=0.0, 
        scaling_modifier = 1.0, 
        require_coord=False, 
        require_depth=True
    )
    return {
        "render": render_pkg["render"],
        "depth": render_pkg["median_depth"],
    }

# Only the parameters of the Gaussians are needed for extracting the mesh.
means = gaussians.get_xyz
scales = gaussians.get_scaling
rotations = gaussians.get_rotation
opacities = gaussians.get_opacity

# Sample Gaussians on the surface.
# Should be performed only once, or just once in a while.
# In this example, we sample at most 600_000 Gaussians.
surface_gaussians_idx = sample_gaussians_on_surface(
    views=train_cameras,
    means=means,
    scales=scales,
    rotations=rotations,
    opacities=opacities,
    n_max_samples=600_000,
    scene_type='outdoor',
)

# Compute initial SDF values for pivots. Should be performed only once.
# In the paper, we propose to learn optimal SDF values by maximizing the 
# consistency between volumetric renderings and surface mesh renderings.
initial_pivots_sdf = compute_initial_sdf_values(
    views=train_cameras,
    render_func=render_func,
    means=means,
    scales=scales,
    rotations=rotations,
    gaussian_idx=surface_gaussians_idx,
)

# Compute Delaunay Triangulation.
# Can be performed once in a while.
delaunay_tets = compute_delaunay_triangulation(
    means=means,
    scales=scales,
    rotations=rotations,
    gaussian_idx=surface_gaussians_idx,
)

# Differentiably extract a mesh from Gaussian parameters, including initial 
# or updated SDF values for the Gaussian pivots.
# This function is differentiable with respect to the parameters of the Gaussians, 
# as well as the SDF values. Can be performed at every training iteration.
mesh = extract_mesh(
    delaunay_tets=delaunay_tets,
    pivots_sdf=initial_pivots_sdf,
    means=means,
    scales=scales,
    rotations=rotations,
    gaussian_idx=surface_gaussians_idx,
)



# You can now apply any differentiable operation on the extracted mesh, 
# and backpropagate gradients back to the Gaussians!
# In the paper, we propose to use differentiable mesh rendering.
from scene.mesh import MeshRasterizer, MeshRenderer
renderer = MeshRenderer(MeshRasterizer(cameras=train_cameras))

# We cull the mesh based on the view frustum for more efficiency
i_view = np.random.randint(0, len(train_cameras))
refined_mesh = frustum_cull_mesh(mesh, train_cameras[i_view])

mesh_render_pkg = renderer(
    refined_mesh, 
    cam_idx=i_view, 
    return_depth=True, return_normals=True
)
mesh_depth = mesh_render_pkg["depth"]
mesh_normals = mesh_render_pkg["normals"]

# 转换为numpy数组后保存
save_dict = {}
for key, value in mesh_render_pkg.items():
    if isinstance(value, torch.Tensor):
        save_dict[key] = value.detach().cpu().numpy()
    else:
        save_dict[key] = value

np.savez("mesh_render_output.npz", **save_dict)

# 保存mesh
# import trimesh

# 从Meshes对象中提取顶点和面
refined_vertices = refined_mesh.verts.detach().cpu().numpy()
refined_faces = refined_mesh.faces.detach().cpu().numpy()

# 创建trimesh对象并保存
refined_mesh_obj = trimesh.Trimesh(vertices=refined_vertices, faces=refined_faces)

# # 保存为OBJ格式
# mesh_obj.export('extracted_mesh.obj')

# 或者保存为PLY格式
refined_mesh_obj.export(f'refined_mesh_{len(fcc_data)}.ply')

vertices = mesh.verts.detach().cpu().numpy()
faces = mesh.faces.detach().cpu().numpy()

# 创建trimesh对象并保存
mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)

# # 保存为OBJ格式
# mesh_obj.export('extracted_mesh.obj')

# 或者保存为PLY格式
mesh_obj.export(f'mesh_{len(fcc_data)}.ply')

# # 或者保存为STL格式
# mesh_obj.export('extracted_mesh.stl')