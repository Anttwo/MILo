import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from tensorf_utils import positional_encoding, N_to_reso, SHRender, MLPRender_Fea, TVLoss, MLPRender_Fea_Diffuse, MLPRender_Fea_Specular

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scene.mesh import Meshes, MeshRenderer
from utils.geometry_utils import transform_points_view_to_world
from utils.sh_utils import eval_sh
from abc import abstractmethod
import trimesh
import math

'''
Mesh Utils
'''

def get_mesh_from_ply(ply_file: str) -> Meshes:
    """
    Loads a mesh from a PLY file and converts it to a our Mesh object.

    Args:
        ply_file (str): The path to the PLY file containing the mesh data.

    Returns:
        Meshes: A Meshes object containing the vertices and faces of the mesh.
    """
    print(f"[INFO] Loading mesh from {ply_file}")
    mesh_data = trimesh.load(ply_file)

    # Convert vertices and faces to PyTorch tensors on GPU
    verts = torch.tensor(mesh_data.vertices, dtype=torch.float32, device="cuda")
    faces = torch.tensor(mesh_data.faces, dtype=torch.int32, device="cuda")
    mesh = Meshes(verts=verts, faces=faces)
    return mesh

def depths_to_points_and_rays_d(view, depthmap1):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = intrins_inv @ points
    points1 = depthmap1.reshape(1,-1) * rays_d
    
    return points1.reshape(3,H,W), rays_d.reshape(3,H,W)

def map_depth_and_normal_to_color(depth_map: torch.Tensor, 
                       normal_map: torch.Tensor, 
                       color_field: torch.nn.Module, 
                       camera_view: torch.Tensor, 
                       epsilon: float = 1e-6,
                       show_progress: bool = False,
                       chunk_size: int = 500_000,
                       max_random_points: int = 1_000_000) -> torch.Tensor:
    """
    Converts a depth map into world space coordinates and retrieves the corresponding color from a color field model.

    Args:
        depth_map (torch.Tensor): Tensor representing the depth map.
        normal_map (torch.Tensor): Tensor representing the normal map in world space.
        color_field_model (torch.nn.Module): Model that maps world coordinates and ray directions to colors.
        camera_view (torch.Tensor): Tensor representing the camera viewpoint.
        show_progress (bool, optional): Flag to enable progress bar display. Defaults to False.
        NOTE: The function can be heavy on memory, so we add chunkify and random sampling mechanisms.
        chunk_size (int, optional): Number of points to process in each chunk. Defaults to 500,000.
        max_random_points (int, optional): Maximum number of random points to sample. Defaults to 1,000,000.

    Returns:
        torch.Tensor: Tensor containing the color information corresponding to the input depth map.
    """
    # Transform depth to world space
    world_coords, ray_directions = depths_to_points_and_rays_d(view=camera_view, depthmap1=depth_map)
    world_coords = world_coords.view(3, -1).permute(1, 0).unsqueeze(0)
    world_coords = transform_points_view_to_world(world_coords, [camera_view]).squeeze(0)

    ray_directions = ray_directions.reshape(3, -1).permute(1, 0).unsqueeze(0) # ray_directions are in view space
    image_plane_coords = transform_points_view_to_world(ray_directions, [camera_view]).squeeze(0) # we retrieve the image plane in world space
    view_directions = world_coords - image_plane_coords
    view_directions_norm = view_directions.norm(dim=-1, keepdim=True)

    view_directions_normalized = view_directions / (view_directions_norm + epsilon) # Normalize ray directions

    normal_map_flat = normal_map.view(-1,3)
    
    
    color_output = torch.zeros((world_coords.size(0), 3), device=world_coords.device)  # Initialize a zero tensor

    random_mask = None
    if max_random_points > 0:
        # Create random mask
        random_mask = torch.randperm(world_coords.size(0))[:max_random_points]
        world_coords = world_coords[random_mask]
        view_directions_normalized = view_directions_normalized[random_mask]
    
    if show_progress:
        print(f"[INFO] Processing color field with chunk size: {chunk_size / 1_000_000}M with shape: {world_coords.shape[0] / 1_000_000}M")
    for i in tqdm(range(0, world_coords.size(0), chunk_size), desc="Processing color field", disable=not show_progress):
        # Get chunks
        coords_chunk = world_coords[i:i + chunk_size]
        directions_chunk = view_directions_normalized[i:i + chunk_size]
        normal_chunk = normal_map_flat[i:i + chunk_size]

        # Get color
        color_chunk = color_field(coords_chunk, view_directions=directions_chunk, normal_directions=normal_chunk)

        if random_mask is None:
            color_output[i:i + chunk_size] = color_chunk
        else:
            color_output[random_mask[i:i + chunk_size]] = color_chunk
    color_output = color_output.permute(1, 0)
    return color_output.view(3, camera_view.image_height, camera_view.image_width)

'''
Color Field Utils
'''

def mipnerf_360_contraction(xyz_sampled: torch.Tensor, t : float = 0.5) -> torch.Tensor:
    """
    Applies a contraction to the input 3D coordinates based on their distance from the origin.
    
    This function contracts points that are far from the origin to bring them closer, while 
    leaving points that are close to the origin unchanged. The contraction is applied such that 
    points with a norm greater than 1 are scaled down.

    Args:
        xyz_sampled (torch.Tensor): 3D coordinates to be contracted (N, 3).
        t (float): The contraction factor.
        
    Returns:
        torch.Tensor: Contracted 3D coordinates (N, 3).
    """
    xyz_contracted = torch.zeros_like(xyz_sampled)
    norm = torch.norm(xyz_sampled, dim=-1, keepdim=True).repeat(1, 3)
    norm_inv = 1.0 / norm
    # Close points
    close_mask = norm <= 1.0
    xyz_contracted[close_mask] = xyz_sampled[close_mask] * t
    # Far away points
    far_away_mask = norm > 1.0
    far_away_value = (2.0 - norm_inv[far_away_mask]) * xyz_sampled[far_away_mask] * norm_inv[far_away_mask]
    xyz_contracted[far_away_mask] = (1.0 - t) * (far_away_value - 1.0) + t

    return xyz_contracted

'''
Color Field
'''

FIELD_TYPES = ["ColorFieldVM"]

class FieldBase(nn.Module):
    def __init__(self, n_voxels=3000**3, device='cuda', use_mlp=False, app_dim=27, app_n_comp=16, **kargs):
        super(FieldBase, self).__init__()
        print(f"[INFO] scene_extent: {kargs['scene_extent']}")
        self.device = device
        self.scene_extent = kargs["scene_extent"]
        self.aabb = - self.scene_extent * torch.ones(3), self.scene_extent * torch.ones(3)
        
        # TensoRF parameters
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.app_dim = app_dim
        self.app_n_comp = app_n_comp
        self.gridSize = N_to_reso(n_voxels, self.aabb)
        self.use_mlp = use_mlp
        print(f"[INFO] gridSize: {self.gridSize}")

    def normalize_coord(self, xyz_sampled: torch.Tensor) -> torch.Tensor:
        """
        Normalizes and contracts 3D coordinates to fit within a specified range.

        This function first normalizes the input coordinates by dividing them by the scene extent.
        It then applies a contraction using the mipnerf_360_contraction function to map the coordinates
        to the range [-2, 2]. Finally, it scales the contracted coordinates to fit within the range [-1, 1].

        Args:
            xyz_sampled (torch.Tensor): The input 3D coordinates to be normalized and contracted.

        Returns:
            torch.Tensor: The normalized and contracted 3D coordinates.
        """
        xyz_normalized = xyz_sampled / self.scene_extent
        xyz_contracted = mipnerf_360_contraction(xyz_normalized, t=0.5) 
        return xyz_contracted

    @abstractmethod
    def define_modules(self):
        pass

    @abstractmethod
    def init_svd_volume(self, res, device):
        pass

    @abstractmethod
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        pass

    @abstractmethod
    def compute_appfeature(self, xyz_sampled):
        pass
    
    @abstractmethod
    def forward(self, xyz_sampled, view_directions, normal_directions):
        pass

class ColorFieldVM(FieldBase):
    def __init__(self, n_voxels=3000**3, device='cuda', use_mlp=False, app_dim=27, app_n_comp=16, **kargs):
        super(ColorFieldVM, self).__init__(n_voxels, device, use_mlp, app_dim, app_n_comp, **kargs)
        self.define_modules()
        self.init_svd_volume(self.gridSize[0], self.device)
        print(f"[WARNING] Model doesn't use normal directions")

    def define_modules(self):
        self.reg = TVLoss()
        if self.use_mlp:
            self.render_module = MLPRender_Fea(inChanel=self.app_dim, viewpe=2, feape=2, featureC=128).to(self.device)
        else:
            self.render_module = SHRender

        print(f"[INFO] Render Model: {self.render_module}")
    
    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.2 * torch.randn((3, self.app_n_comp, res, res), device=device)
        )
        self.line_coef = torch.nn.Parameter(
            0.2 * torch.randn((3, self.app_n_comp, res, 1), device=device)
        )
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False).to(device)
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.render_module, MLPRender_Fea):
            grad_vars += [{'params': self.render_module.parameters(), 'lr': lr_init_network}]
        
        return grad_vars
    
    def compute_appfeature(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        
        return app_features
    
    def TV_loss(self):
        total = 0
        total = total + self.reg(self.plane_coef) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
        return total

    def forward(self, xyz_sampled, view_directions, normal_directions):
        features = self.compute_appfeature(xyz_sampled)
        return self.render_module(xyz_sampled, view_directions, features)
    
        
'''
Render with Color Field is the class to render the mesh with the color field.
'''
class RenderWithColorField(torch.nn.Module):
    def __init__(self, mesh : Meshes, color_field_type : str = "ColorFieldVM", scene_extent : float = 1.0):
        super(RenderWithColorField, self).__init__()
        self.mesh = mesh
        kwargs = {}
        
        kwargs["scene_extent"] = scene_extent
        print(f"[INFO] Initializing color field {color_field_type}")
        color_field = eval(color_field_type)(**kwargs)
        grad_vars = color_field.get_optparam_groups()
        self.color_field = color_field
        self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    def forward(self, viewpoint_idx : int, viewpoint_cam : any, mesh_renderer : MeshRenderer, training : bool = True, n_random_points : int = -1) -> torch.Tensor:
        mesh_render_pkg = mesh_renderer(
                self.mesh,
                cam_idx=viewpoint_idx,
                return_depth=True,
                return_normals=True
        )
        mesh_depth = mesh_render_pkg["depth"].squeeze()
        mesh_normal = mesh_render_pkg["normals"].squeeze()

        mesh_color = map_depth_and_normal_to_color(depth_map=mesh_depth, 
                                        normal_map=mesh_normal, 
                                        color_field=self.color_field, 
                                        camera_view=viewpoint_cam, 
                                        show_progress=False,
                                        chunk_size=500_000,
                                        max_random_points= n_random_points if training else -1)
        return mesh_color