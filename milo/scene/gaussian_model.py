#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.appearance_network import AppearanceNetwork
from utils.sh_utils import SH2RGB
import trimesh

try:
    from diff_gaussian_rasterization_ms import SparseGaussianAdam
except:
    pass
from torch.optim import Adam

def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(
        self, sh_degree : int, 
        use_mip_filter : bool = False, 
        learn_occupancy : bool = False,
        use_radegs_densification : bool = False,
        use_appearance_network : bool = False,
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
        self.use_appearance_network = use_appearance_network
        if use_appearance_network:
            self.appearance_network = AppearanceNetwork(3+64, 3).cuda()        
            self._appearance_embeddings = nn.Parameter(torch.empty(2048, 64).cuda())
            self._appearance_embeddings.data.normal_(0, 1e-4)
        
        self.use_mip_filter = use_mip_filter
        
        # SDF values are actually learned as occupancies between 0 and 1.
        # Occupancies can be converted to truncated SDF values between -1 and 1 with an affine transformation.
        # During training, the base values of occupancies are initialized using depth fusion; 
        # They are not learned but are periodically reset at the beginning of the training.
        # We learn an occupancy shift added to the base occupancies and initialized to 0.
        self.learn_occupancy = learn_occupancy
        self._occupancy_mode = None
        if learn_occupancy:
            self._base_occupancy = torch.empty(0)
            self._occupancy_shift = torch.empty(0)
        
        self.use_radegs_densification = use_radegs_densification
        if self.use_radegs_densification:
            self.xyz_gradient_accum_abs = torch.empty(0)
            self.xyz_gradient_accum_abs_max = torch.empty(0)

    def capture(self):
        to_return = (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
        if self.learn_occupancy:
            to_return += (self._base_occupancy, self._occupancy_shift)
        if self.use_appearance_network:
            to_return += (self.appearance_network.state_dict(), self._appearance_embeddings,)
        return to_return
    
    def restore(self, model_args, training_args):
        start_idx = 12
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args[:start_idx]
        if self.learn_occupancy:
            self._base_occupancy = model_args[start_idx]
            self._occupancy_shift = model_args[start_idx + 1]
            start_idx = start_idx + 2
        if self.use_appearance_network:
            app_dict = model_args[start_idx]
            self._appearance_embeddings = model_args[start_idx + 1]
            start_idx = start_idx + 2
        if start_idx != len(model_args):
            print(f"[ WARNING ] Restoring model with extra arguments: Only {start_idx} arguments expected, but {len(model_args)} provided.")
        
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if self.use_appearance_network:
            self.appearance_network.load_state_dict(app_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        if self.use_mip_filter:
            scales = self.get_scaling
            scales = torch.square(scales) + torch.square(self.filter_3D)
            scales = torch.sqrt(scales)
            return scales
        else:
            return self.get_scaling
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest    
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    # TODO: Add 3D filter
    @property
    def get_opacity_with_3D_filter(self):
        if self.use_mip_filter:
            opacity = self.get_opacity
            scales = self.get_scaling

            # apply 3D filter
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)
            
            scales_after_square = scales_square + torch.square(self.filter_3D) 
            det2 = scales_after_square.prod(dim=1) 
            coef = torch.sqrt(det1 / det2)
            return opacity * coef[..., None]
        else:
            return self.get_opacity
        
    @property
    def get_scaling_n_opacity_with_3D_filter(self):
        if self.use_mip_filter:
            opacity = self.opacity_activation(self._opacity)
            scales = self.get_scaling
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + torch.square(self.filter_3D) 
            det2 = scales_after_square.prod(dim=1) 
            coef = torch.sqrt(det1 / det2)
            scales = torch.sqrt(scales_after_square)
            return scales, opacity * coef[..., None]
        else:
            return self.get_scaling, self.get_opacity
    
    @torch.no_grad()
    def set_occupancy_mode(self, mode: str):
        if not self.learn_occupancy:
            raise ValueError("Occupancy is not learned")
        self._occupancy_mode = mode
    
    @property
    def get_occupancy(self):
        if self.learn_occupancy:
            if self._occupancy_mode is None:
                raise ValueError("Occupancy mode is not set")
            elif self._occupancy_mode == "occupancy_shift":
                return torch.sigmoid(self._base_occupancy + self._occupancy_shift)
            elif self._occupancy_mode == "density_shift":
                raise NotImplementedError("Density shift is not implemented")
            else:
                raise ValueError(f"Unknown occupancy mode: {self._occupancy_mode}")
        else:
            raise ValueError("Occupancy is not learned")
        
    @property
    def get_occupancy_logit(self):
        if self.learn_occupancy:
            if self._occupancy_mode is None:
                raise ValueError("Occupancy mode is not set")
            elif self._occupancy_mode == "occupancy_shift":
                return self._base_occupancy + self._occupancy_shift
            elif self._occupancy_mode == "density_shift":
                raise NotImplementedError("Density shift is not implemented")
            else:
                raise ValueError(f"Unknown occupancy mode: {self._occupancy_mode}")
        else:
            raise ValueError("Occupancy is not learned")
        
    @torch.no_grad()
    def reset_occupancy(self, base_occupancy, gaussian_idx=None, occupancy=None):
        if self.learn_occupancy:
            if self._occupancy_mode is None:
                raise ValueError("Occupancy mode is not set")
            
            # If using occupancy shift
            if self._occupancy_mode == "occupancy_shift":
                if gaussian_idx is None:
                    self._base_occupancy[...] = inverse_sigmoid(base_occupancy)  # (N_gaussians, 9)
                    if occupancy is not None:
                        self._occupancy_shift[...] = inverse_sigmoid(occupancy) - self._base_occupancy
                else:
                    self._base_occupancy[gaussian_idx] = inverse_sigmoid(base_occupancy)  # (N_valid_gaussians, 9)
                    if occupancy is not None:
                        self._occupancy_shift[gaussian_idx] = inverse_sigmoid(occupancy) - self._base_occupancy[gaussian_idx]
            
            # If using density shift
            elif self._occupancy_mode == "density_shift":
                raise NotImplementedError("Density shift is not implemented")

            else:
                raise ValueError(f"Unknown occupancy mode: {self._occupancy_mode}")
        else:
            raise ValueError("Occupancy is not learned")    
    
    def get_appearance_embedding(self, idx):
        return self._appearance_embeddings[idx]
    
    @torch.no_grad()
    def reset_3D_filter(self):
        xyz = self.get_xyz
        self.filter_3D = torch.zeros([xyz.shape[0], 1], device=xyz.device)
        
    def set_mip_filter(self, use_mip_filter: bool):
        self.use_mip_filter = use_mip_filter
        
    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        if not self.use_mip_filter:
            print(f"[ WARNING ] Computing 3D filter but mip filter is not used.")
        
        # print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:
            # focal_x = float(camera.intrinsic[0,0])
            # focal_y = float(camera.intrinsic[1,1])
            W, H = camera.image_width, camera.image_height
            focal_x = W / (2 * math.tan(camera.FoVx / 2.))
            focal_y = H / (2 * math.tan(camera.FoVy / 2.))

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2 # TODO remove hard coded value
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * focal_x + camera.image_width / 2.0
            y = y / z * focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < focal_x:
                focal_length = focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling_with_3D_filter, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            
        if self.learn_occupancy:
            base_occupancy = torch.zeros((self._xyz.shape[0], 9), device="cuda")
            occupancy_shift = torch.zeros((self._xyz.shape[0], 9), device="cuda")
            self._base_occupancy = nn.Parameter(base_occupancy.requires_grad_(False), requires_grad=False)  # Do not learn base occupancy
            self._occupancy_shift = nn.Parameter(occupancy_shift.requires_grad_(True))  # Learn occupancy shift
        
    def _get_tetra_points(
        self, 
        downsample_ratio:float=None, 
        return_sdf_values:bool=False, 
        xyz_idx:torch.Tensor=None, 
        verbose:bool=False,
        scale_points_with_downsample_ratio:bool=True,
        scale_points_factor:float=None,
        opacity_threshold:float=None,
        override_opacity:torch.Tensor=None,
    ):
        """
        Get the tetra points of the Gaussian model.

        Args:
            downsample_ratio (float, optional): The ratio to downsample the tetra points. Defaults to None.
            return_sdf_values (bool, optional): Whether to return the SDF values. Defaults to False.
            xyz_idx (torch.Tensor, optional): The indices of the tetra points to return. 
                If opacity_threshold is provided, xyz_idx should index points that are not filtered out by the opacity threshold,
                such that xyz_idx.max() < (self.get_opacity_with_3D_filter > opacity_threshold).sum().
                Defaults to None. 
                Overrides downsample_ratio if both are provided.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.
            scale_points_with_downsample_ratio (bool, optional): Whether to scale the points with the downsample ratio. 
                Defaults to True. Overrides scale_points_factor if both are provided.
            scale_points_factor (float, optional): The factor to scale the points. Defaults to None.
            opacity_threshold (float, optional): The opacity threshold to filter the points. Defaults to None.
            override_opacity (torch.Tensor, optional): The opacities to use for the tetra points. 
        Raises:
            ValueError: If SDF values are not used but return_sdf_values is True.

        Returns:
            vertices (torch.Tensor): The vertices of the tetra points.
            vertices_scale (torch.Tensor): The scale of the vertices.
            sdf_values (torch.Tensor, optional): The SDF values of the tetra points.
        """
        M = trimesh.creation.box()
        M.vertices *= 2
        
        use_downsample_ratio = (downsample_ratio is not None) and (downsample_ratio < 1.0)
        use_xyz_idx = xyz_idx is not None
        if verbose:
            print(f"[INFO] Downsample ratio: {downsample_ratio}.")
            
        xyz = self.get_xyz
        scale = self.get_scaling_with_3D_filter * 3.
        rots = build_rotation(self._rotation)
        if return_sdf_values:
            if not self.use_sdf_values:
                raise ValueError("SDF values are not used")
            sdf_values = self.get_sdf_values
        
        # Filter points with small opacity
        if (opacity_threshold is not None) and (opacity_threshold > 0.0):
            if override_opacity is not None:
                opacity = override_opacity
                if verbose:
                    print(f"[INFO] Using provided opacity values.")
            else:
                opacity = self.get_opacity_with_3D_filter
            mask = (opacity > opacity_threshold).squeeze()
            xyz = xyz[mask]
            scale = scale[mask]
            rots = rots[mask]
            if return_sdf_values:
                sdf_values = sdf_values[mask]
                
            if verbose:
                print(f"[INFO] Number of tetra points after opacity threshold: {xyz.shape[0]}.")
            
            # Update downsample ratio
            if use_downsample_ratio:
                downsample_ratio = min(downsample_ratio * self._xyz.shape[0] / xyz.shape[0], 1.0)
                use_downsample_ratio = downsample_ratio < 1.0
                if verbose:
                    print(f"[INFO] Updated downsample ratio: {downsample_ratio}.")
        
        if use_downsample_ratio or use_xyz_idx:
            if use_xyz_idx:
                downsample_ratio = xyz_idx.shape[0] / xyz.shape[0]
                if verbose:
                    print(f"[INFO] Using provided xyz_idx to downsample tetra points, with ratio {downsample_ratio}.")
            else:
                xyz_idx = torch.randperm(xyz.shape[0])[:int(xyz.shape[0] * downsample_ratio)]
                if verbose:
                    print(f"[INFO] Downsampling tetra points by {downsample_ratio}.")
                
            xyz = xyz[xyz_idx]
            scale = scale[xyz_idx]
            if scale_points_with_downsample_ratio:
                scale = scale / (downsample_ratio ** (1/3))
            elif scale_points_factor is not None:
                scale = scale * scale_points_factor
            rots = rots[xyz_idx]
            if return_sdf_values:
                sdf_values = sdf_values[xyz_idx]
            if verbose:
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0] * 9}.")
        
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        
        if return_sdf_values:
            return vertices, vertices_scale, sdf_values
        else:
            return vertices, vertices_scale
        
    def get_tetra_points(
        self, 
        let_gradients_flow:bool=False,
        **kwargs
    ):
        if let_gradients_flow:
            return self._get_tetra_points(**kwargs)
        else:
            with torch.no_grad():
                return self._get_tetra_points(**kwargs)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        if self.use_radegs_densification:
            self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
            
        if self.learn_occupancy:
            l.append({'params': [self._occupancy_shift], 'lr': training_args.opacity_lr, "name": "occupancy_shift"})
            
        if self.use_appearance_network:
            l = l + [
                {'params': [self._appearance_embeddings], 'lr': training_args.appearance_embeddings_lr, "name": "appearance_embeddings"},
                {'params': self.appearance_network.parameters(), 'lr': training_args.appearance_network_lr, "name": "appearance_network"}
            ]

        if self.use_appearance_network:
            self.optimizer = Adam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_mip_filter:
            l.append('filter_3D')
        if self.learn_occupancy:
            for i in range(self._base_occupancy.shape[1]):
                l.append('base_occupancy_{}'.format(i))
            for i in range(self._occupancy_shift.shape[1]):
                l.append('occupancy_shift_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        if self.use_mip_filter:
            filter_3D = self.filter_3D.detach().cpu().numpy()
            
        if self.learn_occupancy:
            base_occupancy = self._base_occupancy.detach().cpu().numpy()
            occupancy_shift = self._occupancy_shift.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        to_concatenate = (xyz, normals, f_dc, f_rest, opacities, scale, rotation)
        if self.use_mip_filter:
            to_concatenate = to_concatenate + (filter_3D,)
        if self.learn_occupancy:
            to_concatenate = to_concatenate + (base_occupancy, occupancy_shift)
        attributes = np.concatenate(to_concatenate, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_subset_ply(self, path, indices):
        if indices is None:
            raise ValueError("indices must be provided to save_subset_ply.")

        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.long, device=self._xyz.device)
        else:
            indices = indices.to(self._xyz.device, dtype=torch.long)

        mkdir_p(os.path.dirname(path))

        if indices.numel() == 0:
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
            elements = np.empty(0, dtype=dtype_full)
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
            return

        xyz = self._xyz[indices].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc[indices]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest[indices]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity[indices].detach().cpu().numpy()
        scale = self._scaling[indices].detach().cpu().numpy()
        rotation = self._rotation[indices].detach().cpu().numpy()

        if self.use_mip_filter:
            filter_3D = self.filter_3D[indices].detach().cpu().numpy()

        if self.learn_occupancy:
            base_occupancy = self._base_occupancy[indices].detach().cpu().numpy()
            occupancy_shift = self._occupancy_shift[indices].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        to_concatenate = (xyz, normals, f_dc, f_rest, opacities, scale, rotation)
        if self.use_mip_filter:
            to_concatenate = to_concatenate + (filter_3D,)
        if self.learn_occupancy:
            to_concatenate = to_concatenate + (base_occupancy, occupancy_shift)
        attributes = np.concatenate(to_concatenate, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        if self.use_mip_filter:
            # reset opacity to by considering 3D filter
            current_opacity_with_filter = self.get_opacity_with_3D_filter
            opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
            
            # apply 3D filter
            scales = self.get_scaling
            
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)
            
            scales_after_square = scales_square + torch.square(self.filter_3D) 
            det2 = scales_after_square.prod(dim=1) 
            coef = torch.sqrt(det1 / det2)
            opacities_new = opacities_new / coef[..., None]
            opacities_new = self.inverse_opacity_activation(opacities_new)

            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
        else:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
        
        if self.learn_occupancy:
            # TODO: Values should be properly reset with optimizer
            pass

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        if "filter_3D" in plydata.elements[0]:
            print(f"[INFO] Loading 3D Mip Filter from ply file.")
            self.use_mip_filter = True
            filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]
        else:
            print(f"[INFO] No 3D Mip Filter found in ply file.")
            self.use_mip_filter = False

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        if self.max_sh_degree is None:
            self.max_sh_degree = int(np.sqrt(len(extra_f_names) / 3 + 1) - 1)
        else:
            expected_extra = 3 * (self.max_sh_degree + 1) ** 2 - 3
            if len(extra_f_names) != expected_extra:
                inferred_degree = int(np.sqrt(len(extra_f_names) / 3 + 1) - 1)
                print(
                    f"[WARNING] Expected SH degree {self.max_sh_degree} (extra features {expected_extra}), "
                    f"but file provides {len(extra_f_names)} extra features. "
                    f"Adjusting SH degree to {inferred_degree}."
                )
                self.max_sh_degree = inferred_degree
                expected_extra = 3 * (self.max_sh_degree + 1) ** 2 - 3
                if len(extra_f_names) != expected_extra:
                    raise ValueError(
                        f"Cannot reconcile SH degree. File has {len(extra_f_names)} extra features, "
                        f"which does not match expected {expected_extra} for degree {self.max_sh_degree}."
                    )
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        base_occupancy_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("base_occupancy_")]
        occupancy_shift_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("occupancy_shift_")]
        if len(base_occupancy_names) > 0:
            print(f"[INFO] Loading base occupancy from ply file.")
            self.learn_occupancy = True
            base_occupancy_names = sorted(base_occupancy_names, key = lambda x: int(x.split('_')[-1]))
            occupancy_shift_names = sorted(occupancy_shift_names, key = lambda x: int(x.split('_')[-1]))
            base_occupancy = np.zeros((xyz.shape[0], len(base_occupancy_names)))
            occupancy_shift = np.zeros((xyz.shape[0], len(occupancy_shift_names)))
            for idx, attr_name in enumerate(base_occupancy_names):
                base_occupancy[:, idx] = np.asarray(plydata.elements[0][attr_name])
            for idx, attr_name in enumerate(occupancy_shift_names):
                occupancy_shift[:, idx] = np.asarray(plydata.elements[0][attr_name])
            assert base_occupancy.shape == occupancy_shift.shape
        else:
            print(f"[INFO] No base occupancy found in ply file.")
            self.learn_occupancy = False

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.use_mip_filter:
            self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")
        if self.learn_occupancy:
            self._base_occupancy = nn.Parameter(torch.tensor(base_occupancy, dtype=torch.float, device="cuda").requires_grad_(False), requires_grad=False)
            self._occupancy_shift = nn.Parameter(torch.tensor(occupancy_shift, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.learn_occupancy:
            with torch.no_grad():
                self._base_occupancy = nn.Parameter(self._base_occupancy[valid_points_mask].requires_grad_(False))
            self._occupancy_shift = optimizable_tensors["occupancy_shift"]
            
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        if self.use_radegs_densification:
            self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
            self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._culling = self._culling[valid_points_mask]
        self.factor_culling = self.factor_culling[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 
        new_base_occupancy=None, new_occupancy_shift=None,
    ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        
        if self.learn_occupancy:
            if (new_base_occupancy is None) or (new_occupancy_shift is None):
                raise ValueError("new_base_occupancy and new_occupancy_shift are required when learn_occupancy is True.")
            d["occupancy_shift"] = new_occupancy_shift

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.learn_occupancy:
            self._base_occupancy = torch.cat((self._base_occupancy, new_base_occupancy), dim=0)  # Do not require grad
            self._occupancy_shift = optimizable_tensors["occupancy_shift"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.use_radegs_densification:
            self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def add_densification_stats_culling(self, viewspace_point_tensor, update_filter, factor):
        self.xyz_gradient_accum[update_filter] += (torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)*factor[update_filter])
        self.denom[update_filter] += 1        


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
            
        if self.learn_occupancy:
            new_base_occupancy = self._base_occupancy[selected_pts_mask]
            new_occupancy_shift = self._occupancy_shift[selected_pts_mask]
        else:
            new_base_occupancy = None
            new_occupancy_shift = None

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 
            new_base_occupancy, new_occupancy_shift
        )

        new_culling = self._culling[selected_pts_mask]
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask]
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            
        if self.learn_occupancy:
            new_base_occupancy = self._base_occupancy[selected_pts_mask].repeat(N,1)
            new_occupancy_shift = self._occupancy_shift[selected_pts_mask].repeat(N,1)
        else:
            new_base_occupancy = None
            new_occupancy_shift = None

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, 
            new_base_occupancy, new_occupancy_shift
        )

        new_culling = self._culling[selected_pts_mask].repeat(N,1)
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask].repeat(N,1)
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))                  

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)



    def densify_and_prune_mask(self, max_grad, min_opacity, extent, max_screen_size, mask_split):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_mask(grads, max_grad, extent, mask_split)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()



    def densify_and_split_mask(self, grads, grad_threshold, scene_extent, mask, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        padded_mask = torch.zeros((n_init_points), dtype=torch.bool, device='cuda')
        padded_mask[:grads.shape[0]] = mask
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_mask)
        

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            
        if self.learn_occupancy:
            new_base_occupancy = self._base_occupancy[selected_pts_mask].repeat(N,1)
            new_occupancy_shift = self._occupancy_shift[selected_pts_mask].repeat(N,1)
        else:
            new_base_occupancy = None
            new_occupancy_shift = None

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, 
            new_base_occupancy, new_occupancy_shift
        )

        new_culling = self._culling[selected_pts_mask].repeat(N,1)
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask].repeat(N,1)
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))          

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def reinitial_pts(self, pts, rgb):

        fused_point_cloud = pts
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
            
        if self.learn_occupancy:
            base_occupancy = torch.zeros((fused_point_cloud.shape[0], 9), dtype=torch.float, device="cuda")
            occupancy_shift = torch.zeros((fused_point_cloud.shape[0], 9), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  
        if self.learn_occupancy:
            self._base_occupancy = nn.Parameter(base_occupancy.requires_grad_(False))
            self._occupancy_shift = nn.Parameter(occupancy_shift.requires_grad_(True))

    def init_culling(self, num_views):
        self._culling=torch.zeros((self._xyz.shape[0], num_views), dtype=torch.bool, device='cuda')
        self.factor_culling=torch.ones((self._xyz.shape[0],1), device='cuda')

    def depth_reinit(self, scene, render_depth, iteration, num_depth, args, pipe, background):

        out_pts_list=[]
        gt_list=[]
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
        for view in views:
            gt = view.original_image[0:3, :, :]

            render_depth_pkg = render_depth(view, self, pipe, background, culling=self._culling[:,view.uid])


            out_pts = render_depth_pkg["out_pts"]
            accum_alpha = render_depth_pkg["accum_alpha"]


            prob=1-accum_alpha

            prob = prob/prob.sum()
            prob = prob.reshape(-1).cpu().numpy()

            factor=1/(gt.shape[1]*gt.shape[2]*len(views)/num_depth)

            N_xyz=prob.shape[0]
            num_sampled=int(N_xyz*factor)

            indices = np.random.choice(N_xyz, size=num_sampled, 
                                        p=prob,replace=False)
            
            out_pts = out_pts.permute(1,2,0).reshape(-1,3)
            gt = gt.permute(1,2,0).reshape(-1,3)

            out_pts_list.append(out_pts[indices])
            gt_list.append(gt[indices])       


        out_pts_merged=torch.cat(out_pts_list)
        gt_merged=torch.cat(gt_list)

        return out_pts_merged, gt_merged
    

    def interesction_sampling(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights
        
        imp_score[accum_area_max==0]=0
        prob = imp_score/imp_score.sum()
        prob = prob.cpu().numpy()


        factor=args.sampling_factor
        N_xyz=self._xyz.shape[0]
        num_sampled=int(N_xyz*factor*((prob!=0).sum()/prob.shape[0]))
        indices = np.random.choice(N_xyz, size=num_sampled, 
                                    p=prob, replace=False)

        mask = np.zeros(N_xyz, dtype=bool)
        mask[indices] = True

        self.prune_points(mask==False)

        return self._xyz, SH2RGB(self._features_dc+0)[:,0]


    def interesction_preserving(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0 
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights
            
        imp_score[accum_area_max==0]=0
        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 
        self.prune_points(non_prune_mask==False)

        return self._xyz, SH2RGB(self._features_dc+0)[:,0]


    def importance_pruning(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]

            imp_score=imp_score+accum_weights
            
        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 
        self.prune_points(non_prune_mask==False)

        return self._xyz, SH2RGB(self._features_dc+0)[:,0]


    

    def visibility_culling(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)

            self._culling[:,view.uid]=(non_prune_mask==False)

            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1

            imp_score=imp_score+accum_weights

        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.999) 

        self.factor_culling=count_vis/(count_rad+1e-1)

        mask = (count_vis<=1)[:,0]
        mask = torch.logical_or(mask, non_prune_mask==False)
        self.prune_points(mask) 


    def aggressive_clone(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        for view in views:
            # render_pkg = render_simp(view, self, pipe, background)
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])

            accum_weights = render_pkg["accum_weights"]
            area_max = render_pkg["area_max"]

            imp_score=imp_score+accum_weights
            accum_area_max = accum_area_max+area_max

        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.999) 
        self.prune_points(~non_prune_mask) 

        imp_score[accum_area_max==0]=0
        intersection_pts_mask = init_cdf_mask(importance=imp_score, thres=0.99)
        intersection_pts_mask=intersection_pts_mask[non_prune_mask]
        self.clone(intersection_pts_mask)


    # aggressive_clone with visibility_culling
    def culling_with_clone(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            # render_pkg = render_simp(view, self, pipe, background)
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])

            accum_weights = render_pkg["accum_weights"]
            area_max = render_pkg["area_max"]

            imp_score=imp_score+accum_weights
            accum_area_max = accum_area_max+area_max

            visibility_mask = init_cdf_mask(importance=accum_weights, thres=0.99)
            self._culling[:,view.uid]=(visibility_mask==False)
            count_rad[render_pkg["radii"]>0] += 1
            count_vis[visibility_mask] += 1

        self.factor_culling=count_vis/(count_rad+1e-1)

        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.999) 
        prune_mask = (count_vis==0)[:,0]
        prune_mask = torch.logical_or(prune_mask, non_prune_mask==False)
        self.prune_points(prune_mask) 


        imp_score[accum_area_max==0]=0
        intersection_pts_mask = init_cdf_mask(importance=imp_score, thres=0.99)

        intersection_pts_mask=intersection_pts_mask[~prune_mask]
        self.clone(intersection_pts_mask)


    def clone(self, selected_pts_mask):
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        temp_opacity_old = self.get_opacity[selected_pts_mask]
        new_opacity = 1-(1-temp_opacity_old)**0.5

        temp_scale_old = self.get_scaling[selected_pts_mask]
        new_scaling = (temp_opacity_old / (2*new_opacity-0.5**0.5*new_opacity**2)) * temp_scale_old

        new_opacity = torch.clamp(new_opacity, max=1.0 - torch.finfo(torch.float32).eps, min=0.0051)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling)   


        self._opacity[selected_pts_mask] = new_opacity
        self._scaling[selected_pts_mask] = new_scaling


        new_rotation = self._rotation[selected_pts_mask]
            
        if self.learn_occupancy:
            new_base_occupancy = self._base_occupancy[selected_pts_mask]
            new_occupancy_shift = self._occupancy_shift[selected_pts_mask]
        else:
            new_base_occupancy = None
            new_occupancy_shift = None

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, 
            new_base_occupancy, new_occupancy_shift
        )

        new_culling = self._culling[selected_pts_mask]
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask]
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))
    

    # interesction_preserving with visibility_culling
    def culling_with_interesction_preserving(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0 
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights            

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)

            self._culling[:,view.uid]=(non_prune_mask==False)

            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1


        imp_score[accum_area_max==0]=0
        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 

        self.factor_culling=count_vis/(count_rad+1e-1)


        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, non_prune_mask==False)
        self.prune_points(prune_mask) 


    # interesction_sampling with visibility_culling
    def culling_with_interesction_sampling(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0 
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)

            self._culling[:,view.uid]=(non_prune_mask==False)

            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1


        imp_score[accum_area_max==0]=0
        prob = imp_score/imp_score.sum()
        prob = prob.cpu().numpy()

        factor=args.sampling_factor
        N_xyz=self._xyz.shape[0]
        num_sampled=int(N_xyz*factor*((prob!=0).sum()/prob.shape[0]))
        indices = np.random.choice(N_xyz, size=num_sampled, 
                                    p=prob, replace=False)

        non_prune_mask = np.zeros(N_xyz, dtype=bool)
        non_prune_mask[indices] = True


        self.factor_culling=count_vis/(count_rad+1e-1)

        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, torch.tensor(non_prune_mask==False, device='cuda'))
        self.prune_points(prune_mask) 


    # importance_pruning with visibility_culling
    def culling_with_importance_pruning(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]

            imp_score=imp_score+accum_weights      

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)

            self._culling[:,view.uid]=(non_prune_mask==False)

            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1


        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 

        self.factor_culling=count_vis/(count_rad+1e-1)

        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, non_prune_mask==False)
        self.prune_points(prune_mask) 


    def extend_features_rest(self):

        features = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

    @torch.no_grad()
    def sample_surface_gaussians(self, scene, render_simp, iteration, args, pipe, background, n_samples, sampling_mask=None):
        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0 
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)

            self._culling[:,view.uid]=(non_prune_mask==False)

            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1

        # Probability of each Gaussian to be sampled
        imp_score[accum_area_max==0]=0
        if sampling_mask is not None:
            imp_score[~sampling_mask] = 0.0
        
        prob = imp_score/imp_score.sum()
        prob = prob.cpu().numpy()

        N_xyz=self._xyz.shape[0]
        N_nonzero_prob = (prob !=0 ).sum()
        
        # factor=args.sampling_factor
        # num_sampled=int(N_xyz*factor)
        num_sampled = min(n_samples, N_nonzero_prob)
        
        indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)

        non_prune_mask = np.zeros(N_xyz, dtype=bool)
        non_prune_mask[indices] = True

        self.factor_culling=count_vis/(count_rad+1e-1)

        # Non-sampled Gaussians
        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, torch.tensor(non_prune_mask==False, device='cuda'))
        
        # Sampled Gaussians
        sampled_idx = torch.arange(N_xyz, device='cuda')[~prune_mask]
        return sampled_idx
    
    @torch.no_grad()
    def sample_opacity_gaussians(self, n_samples, sampling_mask=None):
        opacity = self.get_opacity_with_3D_filter.squeeze()
        
        if sampling_mask is not None:
            opacity[~sampling_mask] = 0.0
        
        prob = opacity / opacity.sum()
        prob = prob.cpu().numpy()
        
        N_xyz=self._xyz.shape[0]
        N_nonzero_prob = (prob !=0 ).sum()
        
        num_sampled = min(n_samples, N_nonzero_prob)
        
        indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)

        if False:
            non_prune_mask = np.zeros(N_xyz, dtype=bool)
            non_prune_mask[indices] = True
            sampled_idx = torch.arange(N_xyz, device='cuda')[non_prune_mask]
        else:
            sampled_idx = torch.tensor(indices, device=opacity.device)

        return sampled_idx

    # ------------------RaDe-GS Densification------------------
    def densify_and_split_radegs(
        self, grads, grad_threshold,  
        grads_abs, grad_abs_threshold, 
        scene_extent, N=2
    ):
        assert self.use_radegs_densification
        
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            
        if self.learn_occupancy:
            new_base_occupancy = self._base_occupancy[selected_pts_mask].repeat(N,1)
            new_occupancy_shift = self._occupancy_shift[selected_pts_mask].repeat(N,1)
        else:
            new_base_occupancy = None
            new_occupancy_shift = None
        
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, 
            new_base_occupancy, new_occupancy_shift
        )
        
        new_culling = self._culling[selected_pts_mask].repeat(N,1)
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask].repeat(N,1)
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))      

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        
    def densify_and_clone_radegs(
        self, grads, grad_threshold,  
        grads_abs, grad_abs_threshold, scene_extent
    ):
        assert self.use_radegs_densification
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        # new_opacities = 1-torch.sqrt(1-self.get_opacity[selected_pts_mask]*0.5)
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
            
        if self.learn_occupancy:
            new_base_occupancy = self._base_occupancy[selected_pts_mask]
            new_occupancy_shift = self._occupancy_shift[selected_pts_mask]
        else:
            new_base_occupancy = None
            new_occupancy_shift = None

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 
            new_base_occupancy, new_occupancy_shift
        )

        new_culling = self._culling[selected_pts_mask]
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask]
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))
        
    # use the same densification strategy as GOF https://github.com/autonomousvision/gaussian-opacity-fields
    def densify_and_prune_radegs(
        self, max_grad, min_opacity, extent, max_screen_size
    ):
        assert self.use_radegs_densification
        
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        before = self._xyz.shape[0]
        self.densify_and_clone_radegs(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]

        self.densify_and_split_radegs(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]
        torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune
    
    def add_densification_stats_radegs(
        self, viewspace_point_tensor, update_filter
    ):
        assert self.use_radegs_densification
        
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1
