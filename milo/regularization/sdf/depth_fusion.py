from typing import List, Callable, Union
import torch
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.mesh import Meshes, MeshRasterizer, MeshRenderer, ScalableMeshRenderer
from arguments import PipelineParams
from utils.camera_utils import get_cameras_spatial_extent
from utils.geometry_utils import is_in_view_frustum
from tqdm import tqdm

# TODO: Import the two following functions from utils.geometry_utils
def transform_points_world_to_view(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from world space to view space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    world_view_transforms = torch.stack([camera.world_view_transform for camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        view_points = factors * view_points  # (n_cameras, N, 3)
    return view_points


def transform_points_to_pixel_space(
        points:torch.Tensor,
        cameras:List[Camera],
        points_are_already_in_view_space:bool=False,
        use_p3d_convention:bool=False,
        znear:float=1e-6,
        keep_float:bool=False,
    ):
        """Transform points from world space (3 coordinates) to pixel space (2 coordinates).

        Args:
            points (torch.Tensor): Should have shape (n_cameras, N, 3).
            cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
            points_are_already_in_view_space (bool, optional): Defaults to False.
            use_p3d_convention (bool, optional): Defaults to False.
            znear (float, optional): Defaults to 1e-6.

        Returns:
            torch.Tensor: Has shape (n_cameras, N, 2). 
                In pixel space, (0, 0) is the center of the left-top pixel,
                and (W-1, H-1) is the center of the right-bottom pixel.
        """
        if points_are_already_in_view_space:
            full_proj_transforms = torch.stack([camera.projection_matrix for camera in cameras])  # (n_depth, 4, 4)
            if use_p3d_convention:
                points = torch.tensor([[[-1, -1, 1]]], device=points.device) * points
        else:
            full_proj_transforms = torch.stack([camera.full_proj_transform for camera in cameras])  # (n_cameras, 4, 4)
        
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
        proj_points = points_h @ full_proj_transforms  # (n_cameras, N, 4)
        proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (n_cameras, N, 2)
        # proj_points is currently in a normalized space where 
        # (-1, -1) is the left-top corner of the left-top pixel,
        # and (1, 1) is the right-bottom corner of the right-bottom pixel.

        # For converting to pixel space, we need to scale and shift the normalized coordinates
        # such that (-1/2, -1/2) is the left-top corner of the left-top pixel, 
        # and (H-1/2, W-1/2) is the right-bottom corner of the right-bottom pixel.
        
        height, width = cameras[0].image_height, cameras[0].image_width
        image_size = torch.tensor([[width, height]], device=points.device)
        
        # proj_points = (1. + proj_points) * image_size / 2
        proj_points = (1. + proj_points) / 2 * image_size - 1./2.

        if keep_float:
            return proj_points        
        else:
            return torch.round(proj_points).long()


def get_interpolated_value_from_pixel_coordinates(
    value_img:torch.Tensor,
    pix_coords:torch.Tensor,
    interpolation_mode:str='bilinear',
    padding_mode:str='border',
    align_corners:bool=True,
):
    """
    Get value for pixel coordinates, by interpolating the value_img.

    Args:
        value_img (torch.Tensor): Has shape (H, W, C).
        pix_coords (torch.Tensor): Has shape (N, 2).
        interpolation_mode (str, optional): Defaults to 'bilinear'.
        padding_mode (str, optional): Defaults to 'border'.
        align_corners (bool, optional): Defaults to True.
        
    Returns:
        torch.Tensor: Has shape (N, C).
    """
    height, width = value_img.shape[:2]
    n_points = pix_coords.shape[0]
    
    # Scale and shift pixel coordinates to the range [-1, 1]
    factors = 0.5 * torch.tensor([[width-1, height-1]], dtype=torch.float32).to(pix_coords.device)  # (1, 2)
    scaled_pix_coords = pix_coords / factors - 1.  # (N, 2)
    scaled_pix_coords = scaled_pix_coords.view(1, -1, 1, 2)  # (1, N, 1, 2)

    # Interpolate the value
    interpolated_value = torch.nn.functional.grid_sample(
        input=value_img.permute(2, 0, 1)[None],  # (1, C, H, W)
        grid=scaled_pix_coords,  # (1, N, 1, 2)
        mode=interpolation_mode,
        padding_mode=padding_mode,  # 'reflection', 'zeros'
        align_corners=align_corners,
    )  # (1, C, N, 1)
    
    # Reshape to (N, C)
    interpolated_value = interpolated_value.reshape(-1, n_points).permute(1, 0)
    return interpolated_value


# def get_tsdf_from_single_view(
#     points:torch.Tensor,
#     depth:torch.Tensor,
#     camera:Camera,
#     znear:float=None,
#     zfar:float=None,
#     trunc_margin:float=None,
#     interpolate_depth:bool=True,
#     interpolation_mode:str='bilinear',
#     padding_mode:str='border',
#     align_corners:bool=True,
# ):
#     """
#     Get TSDF values from a single view.
    
#     Args:
#         points (torch.Tensor): Should have shape (N, 3).
#         depth_img (torch.Tensor): Should have shape (H, W), (1, H, W) or (H, W, 1).
#         camera (Camera): Camera.
#         znear (float, optional): Defaults to None. If None, use camera.znear.
#         zfar (float, optional): Defaults to None. If None, use camera.zfar.
#         trunc_margin (float, optional): Defaults to None. If None, no truncation.
#         interpolate_depth (bool, optional): Defaults to True. If True, interpolate depth between pixels.
#         interpolation_mode (str, optional): Defaults to 'bilinear'. Interpolation mode for depth.
#         padding_mode (str, optional): Defaults to 'border'. Padding mode for depth.
#         align_corners (bool, optional): Defaults to True. Align corners for depth.
        
#     Returns:
#         sdf (torch.Tensor): Has shape (N,).
#         valid_mask (torch.Tensor): Has shape (N,).
#     """
#     H, W = depth.squeeze().shape
    
#     # Move points to view space
#     view_points = transform_points_world_to_view(
#         points=points[None], 
#         cameras=[camera],
#     )[0]  # (N, 3)
    
#     # Project points to pixel space
#     pix_points = transform_points_to_pixel_space(
#         points=view_points[None],
#         cameras=[camera],
#         points_are_already_in_view_space=True,
#         keep_float=True,
#     )[0]  # (N, 2)
#     int_pix_points = pix_points.round().long()  # (N, 2)
#     pix_x, pix_y, pix_z = pix_points[..., 0], pix_points[..., 1], view_points[..., 2]
#     int_pix_x, int_pix_y = int_pix_points[..., 0], int_pix_points[..., 1]

#     # Remove points outside view frustum and outside depth range
#     valid_mask = (
#         (pix_x >= 0) & (pix_x <= W-1) 
#         & (pix_y >= 0) & (pix_y <= H-1) 
#         & (pix_z > (camera.znear if znear is None else znear)) 
#         & (pix_z < (camera.zfar if zfar is None else zfar))
#     )  # (N,)
    
#     # Compute depth values
#     depth_values = -torch.ones_like(pix_z)
#     if interpolate_depth:
#         depth_values[valid_mask] = get_interpolated_value_from_pixel_coordinates(
#             value_img=depth.squeeze().unsqueeze(-1),
#             pix_coords=pix_points[valid_mask],
#             interpolation_mode=interpolation_mode,
#             padding_mode=padding_mode,
#             align_corners=align_corners,
#         )[..., 0]  # (N,)
#     else:
#         depth_values[valid_mask] = depth[int_pix_y[valid_mask], int_pix_x[valid_mask]]  # (N,)
#     valid_mask = valid_mask & (depth_values > 0.)
        
#     # Compute SDF
#     sdf = depth_values - pix_z  # (N,)
    
#     # If using truncation:
#     if trunc_margin is not None:
#         # Remove points that are too far away behind the surface
#         valid_mask = valid_mask & (sdf >= -trunc_margin)
        
#         # Normalize depth difference
#         sdf = (sdf / trunc_margin).clamp(min=-1., max=1.)
    
#     return sdf, valid_mask


class AdaptiveTSDF:
    def __init__(
        self,
        points:torch.Tensor,
        trunc_margin:float,
        znear:float=None,
        zfar:float=None,
        initial_sdf_value:float=-1.0,
        use_binary_opacity:bool=False,
    ):
        """
        A class for computing a TSDF field from a set of points and a collection of posed depth maps.
        
        Args:
            points (torch.Tensor): Points at which to compute the TSDF. Has shape (N, 3).
            trunc_margin (float): Truncation margin for the TSDF.
            znear (float): Near clipping plane. If not provided, will use camera.znear when integrating.
            zfar (float): Far clipping plane. If not provided, will use camera.zfar when integrating.
            use_binary_opacity (bool): Whether to use a binary opacity field or a TSDF field.
                Please note that the TSDF field can be approximated into a binary opacity field 
                by using a TSDF with softmax weighting and high temperature.
        """
        
        assert trunc_margin >= 0, "Truncation margin must be positive"
        assert points.shape[1] == 3, "Points must have shape (N, 3)"
        assert (znear is None) or (znear > 0), "znear must be positive"
        assert (zfar is None) or (zfar > znear), "zfar must be greater than znear"
        
        self._n_points = points.shape[0]
        self._points = points
        self._trunc_margin = trunc_margin if not use_binary_opacity else 1.0
        self._znear = znear
        self._zfar = zfar

        # Initialize the field values
        self._use_binary_opacity = use_binary_opacity
        if self._use_binary_opacity:
            self._tsdf = torch.ones(self._n_points, 1, device=points.device)
        else:
            self._tsdf = initial_sdf_value * torch.ones(self._n_points, 1, device=points.device)
        self._weights = torch.zeros(self._n_points, 1, device=points.device)
        self._colors = torch.zeros(self._n_points, 3, device=points.device)
        
    @property
    def device(self):
        return self._points.device
    
    def integrate(
        self, 
        img:torch.Tensor, 
        depth:torch.Tensor,
        camera:Camera, 
        obs_weight=1.0,
        override_points:torch.Tensor=None,
        interpolate_depth:bool=True,
        interpolation_mode:str='bilinear',
        padding_mode:str='border',
        align_corners:bool=True,
        weight_by_softmax:bool=False,
        softmax_temperature:float=1.0,
    ):
        """
        Integrate a new observation into the TSDF.
        
        Args:
            img (torch.Tensor): Image. Has shape (H, W, 3) or (3, H, W).
            depth (torch.Tensor): Depth. Has shape (H, W), (H, W, 1) or (1, H, W).
            camera (GSCamera): Camera.
            obs_weight (float): Weight for the observation.
            override_points (torch.Tensor): Points for integration. Has shape (N, 3). If None, will use the points provided in the constructor.
            interpolate_depth (bool): Whether to interpolate the depth.
            interpolation_mode (str): Interpolation mode.
            padding_mode (str): Padding mode for interpolation.
            align_corners (bool): Whether to align corners for interpolation.
            weight_by_softmax (bool): Whether to weight the interpolation by the softmax.
            softmax_temperature (float): Temperature for the softmax.
        """
        
        # Reshape image and depth to (H, W, 3) and (H, W) respectively
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        depth = depth.squeeze()
        H, W = depth.shape
        
        points = self._points if override_points is None else override_points
        assert points.shape[0] == self._n_points, f"Points must have shape ({self._n_points}, 3)"
        
        # Transform points to view space
        view_points = transform_points_world_to_view(
            points=points.view(1, self._n_points, 3),
            cameras=[camera],
        )[0]  # (N, 3)
        
        # Project points to pixel space
        pix_points = transform_points_to_pixel_space(
            points=view_points.view(1, self._n_points, 3),
            cameras=[camera],
            points_are_already_in_view_space=True,
            keep_float=True,
        )[0]  # (N, 2)
        int_pix_points = pix_points.round().long()  # (N, 2)
        pix_x, pix_y, pix_z = pix_points[..., 0], pix_points[..., 1], view_points[..., 2]
        int_pix_x, int_pix_y = int_pix_points[..., 0], int_pix_points[..., 1]
        
        # Remove points outside view frustum and outside depth range
        valid_mask = (
            (pix_x >= 0) & (pix_x <= W-1) 
            & (pix_y >= 0) & (pix_y <= H-1) 
            & (pix_z > (camera.znear if self._znear is None else self._znear)) 
            & (pix_z < (camera.zfar if self._zfar is None else self._zfar))
        )  # (N,)
        
        if valid_mask.sum() > 0:
            # Get depth and image values at pixel locations
            packed_values = torch.cat(
                [
                    -torch.ones(len(valid_mask), 1, device=self.device),  # Depth values
                    torch.zeros(len(valid_mask), 3, device=self.device)  # Image values
                ], 
                dim=-1
            )  # (N, 4)
            if interpolate_depth:
                packed_values[valid_mask] = get_interpolated_value_from_pixel_coordinates(
                    value_img=torch.cat([depth.unsqueeze(-1), img], dim=-1),  # (H, W, 4)
                    pix_coords=pix_points[valid_mask],
                    interpolation_mode=interpolation_mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )  # (N_valid, 4)
            else:
                packed_values[valid_mask] = torch.cat([depth.unsqueeze(-1), img], dim=-1)[int_pix_y[valid_mask], int_pix_x[valid_mask]]  # (N_valid, 4)
            depth_values = packed_values[..., :1]  # (N, 1)
            img_values = packed_values[..., 1:]  # (N, 3)
            valid_mask = valid_mask & (depth_values[..., 0] > 0.)  # (N,)
            
            # Compute distance
            sdf = ((depth_values - pix_z.unsqueeze(-1)) / self._trunc_margin).clamp_max(1.)  # (N, 1)
            # if not self._use_binary_opacity:
            #     valid_mask = valid_mask & (sdf[..., 0] >= -1.)
            valid_mask = valid_mask & (sdf[..., 0] >= -1.)
            
            # Compute observation weight
            _obs_weight = obs_weight
            if weight_by_softmax:
                _obs_weight = _obs_weight * torch.exp(sdf / softmax_temperature)  # (N_valid, 1)
                
            # Update Field Values
            new_weights = self._weights + _obs_weight  # (N, 1)
            if self._use_binary_opacity:
                new_tsdf = torch.minimum(self._tsdf, (sdf < 0.).float())
            else:
                new_tsdf = (self._tsdf * self._weights + sdf * _obs_weight) / new_weights  # (N, 1)        
            new_colors = (self._colors * self._weights + img_values * _obs_weight) / new_weights  # (N, 3)
            new_colors = new_colors.clamp(min=0., max=1.)  # (N, 3)
            
            # Update field values
            new_weights = torch.where(valid_mask.unsqueeze(-1), new_weights, self._weights)  # (N, 1)
            new_tsdf = torch.where(valid_mask.unsqueeze(-1), new_tsdf, self._tsdf)  # (N, 1)
            new_colors = torch.where(valid_mask.unsqueeze(-1), new_colors, self._colors)  # (N, 3)
            self._weights = new_weights.detach()
            self._tsdf = new_tsdf.detach()
            self._colors = new_colors.detach()
        
        else:
            new_weights = self._weights
            new_tsdf = self._tsdf
            new_colors = self._colors

        return {
            "weights": new_weights,
            "tsdf": new_tsdf,
            "colors": new_colors,
        }

    def return_field_values(self):
        output_pkg = {
            "weights": self._weights,
            "tsdf": self._tsdf,
            "colors": self._colors,
        }
        return output_pkg


def _evaluate_sdf_values(
    points:torch.Tensor, 
    views:List[Camera], 
    masks:torch.Tensor, 
    gaussians:GaussianModel, 
    pipeline:PipelineParams, 
    background:torch.Tensor, 
    kernel_size:int, 
    render_func:Callable, 
    return_colors:bool=False,
    trunc_margin:float=None, 
    use_binary_opacity:bool=False,
):
    """Evaluate the TSDF values at the given points using Depth Fusion.

    Args:
        points (torch.Tensor): Points at which to compute the TSDF. Has shape (N, 3).
        views (List[Camera]): List of cameras.
        masks (List[torch.Tensor]): List of masks.
        gaussians (GaussianModel): Gaussian model.
        pipeline (PipelineParams): Pipeline parameters.
        background (torch.Tensor): Background.
        kernel_size (int): Kernel size.
        trunc_margin (float): Truncation margin.
        render_func (Callable): Function that takes as arguments (views, gaussians, pipeline, background, kernel_size)
            and returns a render package with keys "render" and "depth", containing the rendered image and depth map as
            tensors with shape (3, H, W) and (1, H, W) respectively.
        return_colors (bool, optional): Whether to return the colors. Defaults to False.
        
    Returns:
        sdf (torch.Tensor): TSDF values. Has shape (N,).
        colors (torch.Tensor, optional): Colors. Has shape (N, 3).
    """
    
    if trunc_margin is None:
        trunc_margin = 2e-3 * get_cameras_spatial_extent(views)["radius"]
    
    tsdf_volume = AdaptiveTSDF(
        points=points,
        trunc_margin=trunc_margin,
        use_binary_opacity=use_binary_opacity,
    )
    
    for cam_id, view in enumerate(tqdm(views, desc=f"Fusing with trunc margin {trunc_margin:.6f}")):
        render_pkg = render_func(view, gaussians, pipeline, background, kernel_size)
        tsdf_volume.integrate(
            img=render_pkg["render"], 
            depth=render_pkg["depth"],
            camera=view, 
            obs_weight=1.0,
            override_points=None,
            interpolate_depth=True,
            interpolation_mode='bilinear',
            padding_mode='border',
            align_corners=True,
            weight_by_softmax=False,
            softmax_temperature=1.0,
        )

    field_values = tsdf_volume.return_field_values()
    if return_colors:
        return field_values["tsdf"].squeeze(), field_values["colors"]
    return field_values["tsdf"].squeeze()


def evaluate_sdf_values(
    points:torch.Tensor, 
    views:List[Camera], 
    masks:torch.Tensor, 
    gaussians:GaussianModel, 
    pipeline:PipelineParams, 
    background:torch.Tensor, 
    kernel_size:int, 
    return_colors:bool=False,
    trunc_margin:Union[float, None]=None, 
    use_binary_opacity:bool=False,
    render_func:Callable=None,
):
    if render_func is None:
        raise ValueError("render_func must be provided.")
    
    def render_func_(view, gaussians, pipeline, background, kernel_size):
        render_pkg = render_func(view, gaussians, pipeline, background, kernel_size, require_depth=True, require_coord=False)
        return {
            "render": render_pkg["render"],
            "depth": render_pkg["median_depth"],
        }

    return _evaluate_sdf_values(
        points=points, 
        views=views, 
        masks=masks, 
        gaussians=gaussians, 
        pipeline=pipeline, 
        background=background, 
        kernel_size=kernel_size, 
        render_func=render_func_, 
        return_colors=return_colors, 
        trunc_margin=trunc_margin,
        use_binary_opacity=use_binary_opacity,
    )
        
        
def evaluate_mesh_occupancy(
    points:torch.Tensor, 
    views:List[Camera], 
    mesh:Meshes,
    masks:torch.Tensor, 
    return_colors:bool=False,
    use_scalable_renderer:bool=False,
):
    """Evaluate the TSDF values at the given points using Depth Fusion.

    Args:
        points (torch.Tensor): Points at which to compute the TSDF. Has shape (N, 3).
        views (List[Camera]): List of cameras.
        masks (List[torch.Tensor]): List of masks.
        mesh (Meshes): Mesh.
        return_colors (bool, optional): Whether to return the colors. Defaults to False.
        
    Returns:
        sdf (torch.Tensor): TSDF values. Has shape (N,).
        colors (torch.Tensor, optional): Colors. Has shape (N, 3).
    """
    
    tsdf_volume = AdaptiveTSDF(
        points=points,
        trunc_margin=1.0,
        use_binary_opacity=True,
    )
    
    mesh_rasterizer = MeshRasterizer(cameras=views)
    if use_scalable_renderer:
        mesh_renderer = ScalableMeshRenderer(mesh_rasterizer)
    else:
        mesh_renderer = MeshRenderer(mesh_rasterizer)
    
    for cam_id, view in enumerate(tqdm(views, desc=f"Computing occupancy from mesh")):
        faces_mask = is_in_view_frustum(mesh.verts, view)[mesh.faces].any(axis=1)
        if faces_mask.sum() == 0:
            continue
        render_pkg = mesh_renderer(
            Meshes(verts=mesh.verts, faces=mesh.faces[faces_mask]),
            cam_idx=cam_id,
            return_depth=True,
            return_normals=True,
            use_antialiasing=False,
        )
        tsdf_volume.integrate(
            img=view.original_image.to(points.device), 
            depth=render_pkg["depth"],
            camera=view, 
            obs_weight=1.0,
            override_points=None,
            interpolate_depth=False,
            interpolation_mode='bilinear',
            padding_mode='border',
            align_corners=True,
            weight_by_softmax=False,
            softmax_temperature=1.0,
        )

    field_values = tsdf_volume.return_field_values()
    if return_colors:
        return field_values["tsdf"].squeeze(), field_values["colors"]
    return field_values["tsdf"].squeeze()


def evaluate_mesh_colors(
    # gaussians:GaussianModel,
    views:List[Camera], 
    mesh:Meshes,
    masks:torch.Tensor, 
    use_scalable_renderer:bool=False,
    trunc_margin:float=None,
    override_points:torch.Tensor=None,
    return_hit_mask:bool=False,
) -> torch.Tensor:
    """Evaluate vertex colors using Depth Fusion.
    Vertices that are not visible won't be assigned a color.

    Args:
        views (List[Camera]): List of cameras.
        mesh (Meshes): Mesh.
        masks (List[torch.Tensor]): List of masks.
        use_scalable_renderer (bool, optional): Whether to use the scalable renderer. Defaults to False.
        trunc_margin (float, optional): Truncation margin. Defaults to None.
        override_points (torch.Tensor, optional): Points to override. Defaults to None.
        return_hit_mask (bool, optional): Whether to return the hit mask. Defaults to False.
        
    Returns:
        colors (torch.Tensor): Colors. Has shape (N, 3).
        hit_mask (torch.Tensor, optional): Hit mask. Has shape (N,).
    """
    if trunc_margin is None:
        trunc_margin = 2e-3 * get_cameras_spatial_extent(views)["radius"]

    points = mesh.verts if override_points is None else override_points

    tsdf_volume = AdaptiveTSDF(
        points=points,
        trunc_margin=trunc_margin,
        use_binary_opacity=False,
        initial_sdf_value=-1.1,
    )
    
    mesh_rasterizer = MeshRasterizer(cameras=views)
    if use_scalable_renderer:
        mesh_renderer = ScalableMeshRenderer(mesh_rasterizer)
    else:
        mesh_renderer = MeshRenderer(mesh_rasterizer)
    
    for cam_id, view in enumerate(tqdm(views, desc=f"Computing vertex colors")):
        faces_mask = is_in_view_frustum(mesh.verts, view)[mesh.faces].any(axis=1)
        if faces_mask.sum() == 0:
            continue
        render_pkg = mesh_renderer(
            Meshes(verts=mesh.verts, faces=mesh.faces[faces_mask]),
            cam_idx=cam_id,
            return_depth=True,
            return_normals=True,
            use_antialiasing=False,
        )
        tsdf_volume.integrate(
            img=view.original_image.to(points.device), 
            depth=render_pkg["depth"],
            camera=view, 
            obs_weight=1.0,
            override_points=None,
            interpolate_depth=False,
            interpolation_mode='bilinear',
            padding_mode='border',
            align_corners=True,
            weight_by_softmax=False,
            softmax_temperature=1.0,
        )

    field_values = tsdf_volume.return_field_values()
    if return_hit_mask:
        return field_values["colors"], field_values["tsdf"][..., 0] > -1.1
    return field_values["colors"]


def evaluate_mesh_colors_all_vertices(
    views:List[Camera], 
    mesh:Meshes,
    masks:torch.Tensor, 
    use_scalable_renderer:bool=False,
    trunc_margin:float=None,
) -> torch.Tensor:
    """Evaluate vertex colors for all vertices of the mesh, including the ones that are not visible.
    Useful for avoiding artifacts when rendering the mesh.

    Args:
        views (List[Camera]): List of cameras.
        mesh (Meshes): Mesh.
        masks (torch.Tensor): List of masks.
        use_scalable_renderer (bool, optional): Whether to use the scalable renderer. Defaults to False.
        trunc_margin (float, optional): Truncation margin. Defaults to None.

    Returns:
        vertex_colors (torch.Tensor): Colors. Has shape (N, 3).
    """
    # First pass : evaluate colors for visible vertices
    print(f"[INFO] Computing colors for visible vertices.")
    vertex_colors, color_mask = evaluate_mesh_colors(
        views=views, 
        mesh=mesh,
        masks=masks,
        use_scalable_renderer=use_scalable_renderer,
        trunc_margin=trunc_margin,
        return_hit_mask=True,
    )
    
    # Second pass : evaluate colors for non-visible vertices by using a larger truncation margin
    print(f"[INFO] Estimating colors for non-visible vertices.")
    vertex_colors[~color_mask] = evaluate_mesh_colors(
        views=views,
        mesh=mesh,
        masks=masks,
        use_scalable_renderer=use_scalable_renderer,
        trunc_margin=1.0 * get_cameras_spatial_extent(views)["radius"],
        return_hit_mask=False,
        override_points=mesh.verts[~color_mask],
    )
    return vertex_colors
