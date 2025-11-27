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
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.geometry_utils import transform_points_world_to_view, get_gaussian_normals_from_view, transform_points_to_pixel_space
from utils.general_utils import build_scaling_rotation
from utils.sh_utils import SH2RGB
from .radegs import render_radegs, integrate_radegs
from .gof import render_gof, integrate_gof


def render_old(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_imp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, flag_max_count=True, culling=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None  # TODO: Check if this is correct
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling is None:
        culling = torch.zeros(means3D.shape[0], dtype=torch.bool, device=means3D.device)
    else:
        if culling.dtype != torch.bool:
            culling = culling.to(torch.bool)
        if culling.shape != (means3D.shape[0],):
            culling = culling.reshape(-1)
        if culling.numel() != means3D.shape[0]:
            raise ValueError(f"Culling mask has {culling.numel()} entries, "
                             f"but {means3D.shape[0]} Gaussians provided.")
        if not culling.is_contiguous():
            culling = culling.contiguous()

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_max_count  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        flag_max_count=flag_max_count)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "area_max": accum_max_count,
            }


def render_simp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, culling=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, \
    accum_weights_ptr, accum_weights_count, accum_max_count  = rasterizer.render_simp(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
            }


def render_depth(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, culling=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    res  = rasterizer.render_depth(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    return res


def render_full(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    flag_max_count=True, 
    culling=None,
    compute_expected_normals=False,
    compute_expected_depth=False,
    compute_accurate_median_depth_gradient=False,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None  # TODO: Check if this is correct
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_max_count  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        flag_max_count=flag_max_count)  # Gradient

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth_render_pkg  = rasterizer.render_depth(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)  # No gradient
    
    # Get index per pixel
    gidx = depth_render_pkg['gidx'].squeeze()  # H, W
    
    # Get depth and normals per Gaussian
    gaussians_normals = get_gaussian_normals_from_view(viewpoint_camera, pc, in_view_space=True)  # N_gaussians, 3
    gaussians_depth = transform_points_world_to_view(pc.get_xyz, [viewpoint_camera])[0, ..., 2:]  # N_gaussians, 1
    
    # Get Median depth
    if compute_accurate_median_depth_gradient:
        gaussian_inv_sigmas = build_scaling_rotation(1. / pc.get_scaling_with_3D_filter, pc._rotation)  # N_gaussians, 3, 3
        gaussian_inv_sigmas = gaussian_inv_sigmas @ gaussian_inv_sigmas.transpose(1, 2)  # N_gaussians, 3, 3
        inv_sigmas = gaussian_inv_sigmas[gidx]  # H, W, 3, 3
        centers = pc.get_xyz[gidx]  # H, W, 3
        
        camera_center = viewpoint_camera.camera_center.view(*([1] * (centers.ndim - 1)), 3)  # 1, 1, 3
        rays = torch.nn.functional.normalize(depth_render_pkg['out_pts'].permute(1, 2, 0) - camera_center, dim=-1)  # H, W, 3
        
        rays_T_inv_sigmas = (inv_sigmas @ rays[..., None]).squeeze(-1)  # Equal to r^T * sigma^(-1);  H, W, 3
        depth_points = camera_center + (
            (rays_T_inv_sigmas * (centers - camera_center)).sum(dim=-1, keepdim=True)
            / (rays_T_inv_sigmas * rays).sum(dim=-1, keepdim=True)
        ) * rays  # Corresponds to the "ray-gaussian intersection", i.e. the point with max opacity along the ray;  H, W, 3
        depth = transform_points_world_to_view(depth_points.view(-1, 3), [viewpoint_camera]).view(depth_points.shape)[..., 2][None]  # 1, H, W --Gradient
    
    else:
        idx_depth = gaussians_depth[gidx].permute(2, 0, 1)  # 1, H, W --Gradient
        depth = depth_render_pkg["rendered_depth"] - idx_depth.detach() + idx_depth  # 1, H, W --Gradient
        
    if compute_expected_depth:
        expected_depth, _, _  = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = None,
            culling = culling,
            # colors_precomp = gaussians_depth.repeat(1, 3),
            colors_precomp = torch.cat(
                [gaussians_depth, torch.ones(gaussians_depth.shape[0], 2, device=gaussians_depth.device)],
                dim=-1,
            ),
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            flag_max_count=flag_max_count
        )
        expected_depth, expected_alpha = expected_depth[0:1], expected_depth[1:2]  # 1, H, W --Gradient
    else:
        expected_depth = None
        expected_alpha = None
    
    if compute_expected_normals:
        # TODO: Should we normalize?
        normal_img, _, _  = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = None,
            culling = culling,
            colors_precomp = gaussians_normals,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            flag_max_count=flag_max_count
        )  # 3, H, W --Gradient
    else:
        idx_normals = gaussians_normals[gidx].squeeze(-2).permute(2, 0, 1)  # 3, H, W --Gradient
        normal_img = idx_normals  # 3, H, W --Gradient
    
    # Get other maps
    alpha_img = depth_render_pkg['accum_alpha']  # 1, H, W --No gradient
    out_pts = depth_render_pkg['out_pts']  # 1, H, W --No gradient
    discriminants = depth_render_pkg['discriminants']  # 1, H, W --No gradient

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,  # 3, H, W --Gradient
            "viewspace_points": screenspace_points, 
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,  
            "area_max": accum_max_count,
            "accum_alpha": alpha_img,  # 1, H, W --No gradient
            "out_pts": out_pts,  # 3, H, W --No gradient
            "discriminants": discriminants,  # 1, H, W --No gradient
            "gaussian_idx": gidx,  # 1, H, W --No gradient
            "median_depth": depth,  # 1, H, W --Gradient
            "expected_depth": expected_depth,  # 1, H, W --Gradient
            "normal": normal_img,  # 3, H, W --Gradient
            "expected_alpha": expected_alpha,  # 1, H, W --Gradient
            # TODO: Remove keys below
            "rendered_depth": depth,  # 1, H, W --Gradient  TODO: Remove this
            "rendered_normals": normal_img,  # 3, H, W --Gradient  TODO: Remove this
            }


def render_isosurface(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    lambda_isosurface = 0.5, 
    scaling_modifier = 1.0, 
    override_color = None, 
    flag_max_count=True, 
    culling=None,
    compute_expected_normals=False,
    compute_expected_depth=False,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None  # TODO: Check if this is correct
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_max_count  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        flag_max_count=flag_max_count)  # Gradient

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth_render_pkg  = rasterizer.render_depth(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)  # No gradient
    
    # Get index per pixel
    gidx = depth_render_pkg['gidx'].squeeze()  # H, W
    
    # Get depth and normals per Gaussian
    gaussians_normals = get_gaussian_normals_from_view(viewpoint_camera, pc, in_view_space=True)  # N_gaussians, 3
    gaussians_depth = transform_points_world_to_view(pc.get_xyz, [viewpoint_camera])[0, ..., 2:]  # N_gaussians, 1
    
    # Get isosurface depth
    gaussian_inv_scaled_rot = build_scaling_rotation(1. / pc.get_scaling_with_3D_filter, pc._rotation).transpose(1, 2)  # Equal to S^(-1) @ R^T;  N_gaussians, 3, 3
    inv_scaled_rot = gaussian_inv_scaled_rot[gidx]  # H, W, 3, 3
    centers = pc.get_xyz[gidx]  # H, W, 3
    
    camera_center = viewpoint_camera.camera_center.view(*([1] * (centers.ndim - 1)), 3)  # 1, 1, 3
    rays = torch.nn.functional.normalize(depth_render_pkg['out_pts'].permute(1, 2, 0) - camera_center, dim=-1)  # H, W, 3
    
    transformed_rays = (inv_scaled_rot @ rays[..., None])[..., 0]  # H, W, 3
    transformed_gauss_to_cam = (inv_scaled_rot @ (camera_center - centers)[..., None])[..., 0]  # H, W, 3
    
    pol_a = (transformed_rays * transformed_rays).sum(dim=-1, keepdim=True)  # H, W, 1
    pol_b = 2 * (transformed_rays * transformed_gauss_to_cam).sum(dim=-1, keepdim=True)  # H, W, 1
    pol_c = (transformed_gauss_to_cam * transformed_gauss_to_cam).sum(dim=-1, keepdim=True) + 2 * np.log(lambda_isosurface)  # H, W, 1
    
    pol_discriminant = pol_b * pol_b - 4 * pol_a * pol_c
    iso_mask = pol_discriminant.detach().permute(2, 0, 1) >= 0
    pol_discriminant = pol_discriminant.clamp(min=0)  # If no solution, we return the ray-Gaussian intersection point (the point with max opacity along the ray)
    
    t_star = (-pol_b - torch.sqrt(pol_discriminant)) / (2 * pol_a)  # H, W, 1
    
    depth_points = camera_center + t_star * rays  # Corresponds to the closest isosurface point, i.e. the first point with opacity lambda_isosurface along the ray;  H, W, 3
    iso_depth = transform_points_world_to_view(depth_points.view(-1, 3), [viewpoint_camera]).view(depth_points.shape)[..., 2][None]  # 1, H, W --Gradient
        
    if compute_expected_depth:
        expected_depth, _, _  = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = None,
            culling = culling,
            # colors_precomp = gaussians_depth.repeat(1, 3),
            colors_precomp = torch.cat(
                [gaussians_depth, torch.ones(gaussians_depth.shape[0], 2, device=gaussians_depth.device)],
                dim=-1,
            ),
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            flag_max_count=flag_max_count
        )
        expected_depth, expected_alpha = expected_depth[0:1], expected_depth[1:2]  # 1, H, W --Gradient
    else:
        expected_depth = None
        expected_alpha = None
    
    if compute_expected_normals:
        # TODO: Should we normalize?
        normal_img, _, _  = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = None,
            culling = culling,
            colors_precomp = gaussians_normals,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            flag_max_count=flag_max_count
        )  # 3, H, W --Gradient
    else:
        idx_normals = gaussians_normals[gidx].squeeze(-2).permute(2, 0, 1)  # 3, H, W --Gradient
        normal_img = idx_normals  # 3, H, W --Gradient
    
    # Get other maps
    alpha_img = depth_render_pkg['accum_alpha']  # 1, H, W --No gradient
    out_pts = depth_render_pkg['out_pts']  # 1, H, W --No gradient
    discriminants = depth_render_pkg['discriminants']  # 1, H, W --No gradient

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,  # 3, H, W --Gradient
            "viewspace_points": screenspace_points, 
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,  
            "area_max": accum_max_count,
            "isosurface_depth": iso_depth,  # 1, H, W --Gradient
            "isosurface_mask": iso_mask,  # 1, H, W --No gradient
            "accum_alpha": alpha_img,  # 1, H, W --No gradient
            "out_pts": out_pts,  # 3, H, W --No gradient
            "discriminants": discriminants,  # 1, H, W --No gradient
            "rendered_normals": normal_img,  # 3, H, W --Gradient
            "gaussian_idx": gidx,  # 1, H, W --No gradient
            "expected_depth": expected_depth,  # 1, H, W --Gradient
            "expected_alpha": expected_alpha,  # 1, H, W --Gradient
            }
