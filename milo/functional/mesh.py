from typing import Union
import numpy as np
import torch
from scene.cameras import Camera
from scene.mesh import Meshes
from utils.tetmesh import marching_tetrahedra
from utils.geometry_utils import is_in_view_frustum
from functional.pivots import extract_gaussian_pivots


def frustum_cull_mesh(
    mesh:Meshes,
    viewpoint_cam:Camera,
) -> Meshes:
    """Cull the mesh based on the view frustum of the given viewpoint.

    Args:
        mesh (Meshes): The mesh to cull.
        viewpoint_cam (Camera): The viewpoint camera.

    Returns:
        Meshes: The culled mesh.
    """
    faces_mask = is_in_view_frustum(mesh.verts, viewpoint_cam)[mesh.faces].any(axis=1)
    return Meshes(
        verts=mesh.verts, 
        faces=mesh.faces[faces_mask], 
        verts_colors=mesh.verts_colors
    )


def extract_mesh(
    delaunay_tets:torch.Tensor,
    pivots_sdf:torch.Tensor,
    means:Union[torch.Tensor, None]=None,
    scales:Union[torch.Tensor, None]=None,
    rotations:Union[torch.Tensor, None]=None,
    gaussian_idx:Union[torch.Tensor, None]=None,
    scale_pivots_with_downsample_ratio:bool=True,
    scale_pivots_factor:Union[float, None]=None,
    override_pivots:Union[torch.Tensor, None]=None,
    override_pivots_scale:Union[torch.Tensor, None]=None,
    filter_large_edges:bool=True,
    collapse_large_edges:bool=False,
) -> Meshes:
    """Differentiably extract a mesh from Gaussian parameters, including initial 
    or updated SDF values for the Gaussian pivots.
    This function is differentiable with respect to the parameters of the Gaussians, 
    as well as the SDF values. Can be performed at every training iteration.

    Args:
        delaunay_tets (torch.Tensor): The Delaunay tetrahedra. Shape: (N_tets, 4).
        pivots_sdf (torch.Tensor): The SDF values for the Gaussian pivots. Shape: (N_pivots,).
        means (Union[torch.Tensor, None], optional): The means of the Gaussians. Shape: (N_gaussians, 3). Defaults to None.
        scales (Union[torch.Tensor, None], optional): The scales of the Gaussians. Shape: (N_gaussians, 3). Defaults to None.
        rotations (Union[torch.Tensor, None], optional): The rotations of the Gaussians. Shape: (N_gaussians, 4). Defaults to None.
        gaussian_idx (Union[torch.Tensor, None], optional): The indices of the Gaussians to be used for generating pivots. 
            Shape: (N_selected,). Defaults to None.
        scale_pivots_with_downsample_ratio (bool, optional): If True, the scale of the pivots will be adjusted to match the downsample ratio. Defaults to True.
        scale_pivots_factor (Union[float, None], optional): If provided, the scale of the pivots will be multiplied by this factor. Defaults to None.
        override_pivots (Union[torch.Tensor, None], optional): Override pivots. Shape: (N_pivots, 3). Defaults to None.
        override_pivots_scale (Union[torch.Tensor, None], optional): Override pivots scale. Shape: (N_pivots, 1). Defaults to None.
        filter_large_edges (bool, optional): If True, filter out large edges. Defaults to True.
        collapse_large_edges (bool, optional): If True, collapse large edges onto the pivot with the smallest SDF value. Defaults to False.

    Returns:
        Meshes: The extracted mesh.
    """
    assert (
        (
            (means is not None) and (scales is not None) and (rotations is not None)
        ) or (
            (override_pivots is not None) and (override_pivots_scale is not None)
        )
    )
    
    # Extract pivots from Gaussians.
    # If override_pivots is provided, use it instead of extracting pivots from Gaussians.
    if override_pivots is None:
        pivots, pivots_scale = extract_gaussian_pivots(
            means=means,
            scales=scales,
            rotations=rotations,
            gaussian_idx=gaussian_idx,
            scale_pivots_with_downsample_ratio=scale_pivots_with_downsample_ratio,
            scale_pivots_factor=scale_pivots_factor
        )
    else:
        pivots = override_pivots
        pivots_scale = override_pivots_scale

    # --- Marching Tetrahedra ---
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(
        vertices=pivots[None],
        tets=delaunay_tets,
        sdf=pivots_sdf.reshape(1, -1),
        scales=pivots_scale[None]
    )
    end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
    end_scales = scale_list[0]  # (N_verts, 2, 1)
    
    norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)
    verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]        
    faces = faces_list[0]  # (N_faces, 3)

    # --- Filtering ---
    if filter_large_edges or collapse_large_edges:
        dmtet_distance = torch.norm(end_points[:, 0, :] - end_points[:, 1, :], dim=-1)
        dmtet_scale = end_scales[:, 0, 0] + end_scales[:, 1, 0]
        dmtet_vertex_mask = (dmtet_distance <= dmtet_scale)
    
    # Filtering for large edges, inspired by GOF.
    # If the edge between two pivots is larger than the sum 
    # of the scales of the two corresponding Gaussians,
    # The pivots should probably not be connected.
    if filter_large_edges:
        dmtet_face_mask = dmtet_vertex_mask[faces].all(axis=1)
        faces = faces[dmtet_face_mask]
    
    # The following option collapses big edges
    # onto the pivot with the smallest SDF value.
    if collapse_large_edges:
        min_end_points = end_points[
            np.arange(end_points.shape[0]), 
            end_sdf.argmin(dim=1).flatten().cpu().numpy()
        ]  # TODO: Do the computation only for filtered vertices
        verts = torch.where(dmtet_vertex_mask[:, None], verts, min_end_points)

    # --- Build Mesh ---
    return Meshes(verts=verts, faces=faces)


# TODO: Function to compute mesh colors 
# TODO: Function to compute mesh occupancy labels