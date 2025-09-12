from typing import Callable, Dict, Any, Tuple
from functools import partial
import gc
import numpy as np
import torch
from arguments import PipelineParams
from scene import Scene
from scene.cameras import Camera
from gaussian_renderer import render_simp
from scene.mesh import Meshes, MeshRasterizer, MeshRenderer, ScalableMeshRenderer
from scene.gaussian_model import GaussianModel
from utils.tetmesh import marching_tetrahedra
from utils.camera_utils import get_cameras_spatial_extent
from utils.geometry_utils import is_in_view_frustum
from utils.geometry_utils import depth_to_normal as depth_double_to_normal
from regularization.sdf.integration import (
    evaluate_cull_sdf_values,
)
from regularization.sdf.depth_fusion import (
    evaluate_mesh_occupancy,
)
from regularization.sdf.depth_fusion import evaluate_sdf_values as evaluate_sdf_values_depth_fusion
from regularization.sdf.learnable import (
    compute_initial_sdf_with_binary_search,
    convert_sdf_to_occupancy,
    convert_occupancy_to_sdf,
)
from utils.geometry_utils import (
    unflatten_voronoi_features,
    flatten_voronoi_features,
)

# Try importing cpp extension, handle potential ImportError
try:
    from tetranerf.utils.extension import cpp
except ImportError:
    cpp = None
    print("[WARNING] Could not import 'tetranerf.utils.extension.cpp'. Mesh regularization requires this.")


def initialize_mesh_regularization(
    scene: Scene, 
    config: Dict[str, Any], 
) -> Tuple[MeshRenderer, Dict[str, Any]]:
    """
    Initializes components required for mesh regularization.

    Args:
        scene: The scene object containing training cameras.
        config: Configuration dictionary for mesh regularization.

    Returns:
        A dictionary containing initialized components:
        - mesh_renderer: The initialized MeshRenderer.
        - state: A dictionary to hold mesh regularization state variables.
    """
    if cpp is None:
        raise ImportError("Mesh regularization requires 'tetranerf.utils.extension.cpp'. Please ensure it's compiled.")

    print("[INFO] Mesh regularization enabled.")
    print(f"         > Mesh depth loss type: {config['mesh_depth_loss_type']}")
    print(f"         > Occupancy mode: {config['occupancy_mode']}")
        
    mesh_rasterizer = MeshRasterizer(cameras=scene.getTrainCameras().copy(), use_opengl=config['use_opengl'])
    if config["use_scalable_renderer"]:
        print("[INFO] Using scalable mesh renderer.")
        mesh_renderer = ScalableMeshRenderer(mesh_rasterizer)
    else:
        mesh_renderer = MeshRenderer(mesh_rasterizer)

    # Initialize state dictionary
    mesh_state = {
        "delaunay_tets": None,
        "voronoi_occupancy_labels": None,
        "delaunay_xyz_idx": None,
        "surface_delaunay_xyz_idx": None,
        "reset_delaunay_samples": True,
        "reset_sdf_values": True,
    }

    return mesh_renderer, mesh_state


def compute_mesh_regularization(
    iteration: int,
    render_pkg: Dict[str, torch.Tensor],
    viewpoint_cam: Camera,
    viewpoint_idx: int,
    gaussians: GaussianModel,
    scene: Scene,
    pipe: PipelineParams,
    background: torch.Tensor,
    kernel_size: float,
    config: Dict[str, Any],
    mesh_renderer: MeshRenderer,
    mesh_state: Dict[str, Any],
    render_func: Callable,
    weight_adjustment: float=0.005,
    args: Any=None,
    integrate_func:Callable=None,
) -> Dict[str, Any]:
    """
    Computes the mesh regularization loss and updates the mesh state.

    Args:
        iteration: Current training iteration.
        render_pkg: Dictionary containing rendering results.
            - render: The rendered image.
            - median_depth: The median depth of the rendered image.
            - expected_depth: The expected depth of the rendered image.
        viewpoint_cam: The current viewpoint camera.
        viewpoint_idx: Index of the current viewpoint camera.
        gaussians: The GaussianModel object.
        scene: The scene object.
        pipe: Pipeline parameters.
        background: Background color tensor.
        kernel_size: Kernel size for rendering.
        config: Configuration dictionary for mesh regularization.
        mesh_renderer: The MeshRenderer object.
        mesh_state: Dictionary holding the current state of mesh regularization.
        render_func: Function to render the scene. 
            Takes as input: 
            - A camera viewpoint_camera
            - A GaussianModel pc
            - A pipeline pipe
            - A background color bg_color
            Returns a dictionary with the following keys:
            - render: The rendered image.
            - median_depth: The median depth of the rendered image.

    Returns:
        A dictionary containing:
        - mesh_loss: The computed total mesh regularization loss (torch.Tensor).
        - mesh_depth_loss: The depth component of the mesh loss (torch.Tensor).
        - mesh_normal_loss: The normal component of the mesh loss (torch.Tensor).
        - updated_state: The updated mesh_state dictionary.
        - mesh_render_pkg: Dictionary containing mesh rendering results (depth, normals).
        - voronoi_points_count: Number of points used for Delaunay triangulation.
    """

    lambda_mesh_depth = config["depth_weight"]
    lambda_mesh_normal = config["normal_weight"]

    # --- State Management ---
    # For filtering Delaunay points
    delaunay_xyz_idx = mesh_state["delaunay_xyz_idx"]  # (N_voronoi_Gaussians,)
    # For SDF regularization by occupancy labels
    voronoi_occupancy_labels = mesh_state["voronoi_occupancy_labels"]  # (N_voronoi_points, )
    # Delaunay tetrahedralization
    delaunay_tets = mesh_state["delaunay_tets"]  # (N_tets, 4)
    # Flags
    reset_delaunay_samples = mesh_state["reset_delaunay_samples"]
    reset_sdf_values = mesh_state["reset_sdf_values"]
    # For logging
    voronoi_points_count = 0
    
    # Used for filtering Delaunay points
    use_delaunay_downsampling = (
        (config["n_max_points_in_delaunay"] is not None)
        and (config["n_max_points_in_delaunay"] > 0)
    )

    # Check for resets based on intervals
    if iteration < config['stop_iter']:
        if iteration % config["delaunay_reset_interval"] == 0:
            delaunay_tets = None
            reset_delaunay_samples = True

        if iteration % config["sdf_reset_interval"] == 0:
            reset_sdf_values = True
        
        if config["fix_set_of_learnable_sdfs"] and (iteration > config["start_iter"]):
            reset_delaunay_samples = False
            
        if (config["learnable_sdf_reset_mode"] == "none")  and (iteration > config["start_iter"]):
            reset_sdf_values = False  # TODO: Maybe not needed?
        
        if iteration >= config["learnable_sdf_reset_stop_iter"]:
            assert iteration > config["start_iter"]
            reset_sdf_values = False
    else:
        print(f"[INFO] Stopping mesh regularization at iteration {iteration}.")
        print(f"          > Skipping Delaunay and SDF resets for last iterations.")

    if delaunay_tets is None:
        print(f"[INFO] Resetting Delaunay state at iteration {iteration}.")
    if reset_sdf_values:
        print(f"[INFO] Resetting SDF state at iteration {iteration}.")

    # Start mesh regularization logic
    if iteration == config["start_iter"]:
        print("[INFO] Starting mesh regularization at iteration {}".format(iteration))
        print(f"          > Spatial scale for mesh depth loss: {gaussians.spatial_lr_scale}")
        print(f"          > Use Delaunay downsampling: {use_delaunay_downsampling}")
        print(f"          > Use foreground culling: {config['radius_culling'] > 0.0}")
        assert gaussians.spatial_lr_scale > 0.

        # Force resets on the first iteration
        delaunay_tets = None
        reset_sdf_values = True
        reset_delaunay_samples = True

        gaussians.set_occupancy_mode(config["occupancy_mode"])
        print(f"          > Occupancy mode: {gaussians._occupancy_mode}")
        print(f"          > Filter large edges: {config['filter_large_edges']}")
        print(f"          > Fixing set of learnable SDFs: {config['fix_set_of_learnable_sdfs']}")
        print(f"          > Method to reset SDF: {config['method_to_reset_sdf']}")
        print(f"          > Number of binary steps to reset SDF: {config['n_binary_steps_to_reset_sdf']}")
        print(f"          > Number of linearization steps to reset SDF: {config['sdf_reset_linearization_n_steps']}")
        print(f"          > Learnable SDF reset mode: {config['learnable_sdf_reset_mode']}")
        if config["learnable_sdf_reset_mode"] == "ema":
            print(f"             > Learnable SDF reset alpha EMA: {config['learnable_sdf_reset_alpha_ema']}")
        print(f"          > Enforcing occupied centers: {config['enforce_occupied_centers']}")
        print(f"          > Using occupancy labels loss: {config['use_occupancy_labels_loss']}")
        if config["use_occupancy_labels_loss"]:
            print(f"             > Reset occupancy labels every: {config['reset_occupancy_labels_every']}")
        print(f"          > Initializing SDF values by integrating Gaussian occupancy values...")
        # TODO: Do a reset here?

    # --- Main Mesh Regularization Computation ---
    if iteration >= config["start_iter"]:
        # Reset Delaunay samples if needed
        reset_occupancy_labels_for_new_delaunay_sites = False
        if use_delaunay_downsampling:
            if reset_delaunay_samples:
                n_gaussians_to_sample_from = gaussians._xyz.shape[0]
                    
                if config["radius_culling"] > 0.0:
                    cam_spatial_extent = get_cameras_spatial_extent(scene.getTrainCameras().copy())
                    delaunay_radius = cam_spatial_extent["radius"]
                    delaunay_center = cam_spatial_extent["avg_cam_center"].view(1, 3)            
                    with torch.no_grad():
                        delaunay_sampling_radius_mask = (
                            (gaussians._xyz - delaunay_center).norm(dim=-1) <= delaunay_radius
                        ).view(-1)
                    n_gaussians_to_sample_from = int(delaunay_sampling_radius_mask.sum().item())
                else:
                    delaunay_sampling_radius_mask = None

                n_max_gaussians_for_delaunay = int(config["n_max_points_in_delaunay"] / 9.)
                downsample_gaussians_for_delaunay = n_max_gaussians_for_delaunay < n_gaussians_to_sample_from

                if downsample_gaussians_for_delaunay:
                    print(f"[INFO] Downsampling Delaunay Gaussians from {n_gaussians_to_sample_from} to {n_max_gaussians_for_delaunay}.")                        
                    if config["delaunay_sampling_method"] == "random":
                        delaunay_xyz_idx = torch.randperm(
                            n_gaussians_to_sample_from, device="cuda"
                        )[:n_max_gaussians_for_delaunay]
                    elif config["delaunay_sampling_method"] == "surface":
                        delaunay_xyz_idx = gaussians.sample_surface_gaussians(
                            scene=scene,
                            render_simp=render_simp,
                            iteration=iteration,
                            args=args,
                            pipe=pipe,
                            background=background,
                            n_samples=n_max_gaussians_for_delaunay,
                            sampling_mask=delaunay_sampling_radius_mask,
                        )
                    elif config["delaunay_sampling_method"] == "surface+opacity":
                        delaunay_xyz_idx = gaussians.sample_surface_gaussians(
                            scene=scene,
                            render_simp=render_simp,
                            iteration=iteration,
                            args=args,
                            pipe=pipe,
                            background=background,
                            n_samples=n_max_gaussians_for_delaunay,
                            sampling_mask=delaunay_sampling_radius_mask,
                        )
                        n_remaining_gaussians_to_sample = n_max_gaussians_for_delaunay - delaunay_xyz_idx.shape[0]
                        if n_remaining_gaussians_to_sample > 0:
                            mesh_state["surface_delaunay_xyz_idx"] = delaunay_xyz_idx.clone()
                            opacity_sample_mask = torch.ones(gaussians._xyz.shape[0], device="cuda", dtype=torch.bool)
                            opacity_sample_mask[delaunay_xyz_idx] = False
                            delaunay_xyz_idx = torch.cat(
                                [
                                    delaunay_xyz_idx,
                                    gaussians.sample_opacity_gaussians(
                                        n_samples=n_remaining_gaussians_to_sample,
                                        sampling_mask=opacity_sample_mask,
                                    )
                                ],
                                dim=0,
                            )
                            delaunay_xyz_idx = torch.sort(delaunay_xyz_idx, dim=0)[0]
                    else:
                        raise ValueError(f"Invalid Delaunay sampling method: {config['delaunay_sampling_method']}")
                    print(f"[INFO] Downsampled Delaunay Gaussians from {n_gaussians_to_sample_from} to {len(delaunay_xyz_idx)}.")
                    reset_occupancy_labels_for_new_delaunay_sites = True
                else:
                    if delaunay_sampling_radius_mask is not None:
                        delaunay_xyz_idx = torch.where(delaunay_sampling_radius_mask)[0]
                        print(f"[INFO] Using foreground culling with radius {config['radius_culling']}.")
                    else:
                        delaunay_xyz_idx = None
                        print(f"[INFO] No need to downsample Delaunay Gaussians.")

                torch.cuda.empty_cache()
                reset_delaunay_samples = False # Reset flag after computation
                delaunay_tets = None # If downsampling, we need to recompute the tetrahedra
        else:
            delaunay_xyz_idx = None # Ensure it's None if not used

        # Compute Voronoi generators
        # Pass delaunay_xyz_idx which might be None (use all), or indices after opacity/downsampling
        voronoi_points, voronoi_scale = gaussians.get_tetra_points(
            downsample_ratio=None,
            let_gradients_flow=True,
            xyz_idx=delaunay_xyz_idx, # Pass the computed indices
        )
        voronoi_points_count = voronoi_points.shape[0]
        # Recompute Delaunay tetrahedralization if needed
        if delaunay_tets is None:
            print(f"[INFO] Recomputing Delaunay tetrahedralization for {voronoi_points.shape[0]} points...")
            with torch.no_grad():
                # Ensure points are detached before passing to C++ extension
                delaunay_tets = cpp.triangulate(voronoi_points.detach()).cuda().long()
            torch.cuda.empty_cache()

        # --- Compute SDF values ---
        # Check if an SDF reset has to be enforced because of a shape mismatch
        if not reset_sdf_values:
            n_voronoi_sdf = voronoi_points.shape[0]
            if n_voronoi_sdf != voronoi_points.shape[0]:
                print(f"[WARNING] Delaunay SDFs ({n_voronoi_sdf}) and points ({voronoi_points.shape[0]}) count mismatch. Resetting SDFs.")
                reset_sdf_values = True
        
        if reset_sdf_values:
            with torch.no_grad():
                # Get base occupancy values for all voronoi points
                if config["method_to_reset_sdf"] == 'integration':                        
                    sdf_function = partial(
                        evaluate_cull_sdf_values,
                        views=scene.getTrainCameras().copy(), 
                        masks=None, 
                        gaussians=gaussians, 
                        pipeline=pipe, 
                        background=background, 
                        kernel_size=kernel_size, 
                        return_colors=False, 
                        isosurface_value=config["sdf_default_isosurface"], 
                        transform_sdf_to_linear_space=config["transform_sdf_to_linear_space"], 
                        min_occupancy_value=config["min_occupancy_value"],
                        integrate_func=integrate_func,
                    )
                    
                elif config["method_to_reset_sdf"] == 'depth_fusion':                        
                    sdf_function = partial(
                        evaluate_sdf_values_depth_fusion,
                        views=scene.getTrainCameras().copy(), 
                        masks=None, 
                        gaussians=gaussians, 
                        pipeline=pipe, 
                        background=background, 
                        kernel_size=kernel_size, 
                        return_colors=False,
                        trunc_margin=None, 
                        render_func=render_func,
                    )
                
                # Compute and linearize initial occupancy values with binary search if needed
                base_occupancy = compute_initial_sdf_with_binary_search(
                    voronoi_points=voronoi_points,
                    voronoi_scales=voronoi_scale,
                    delaunay_tets=delaunay_tets,
                    sdf_function=sdf_function,
                    n_binary_steps=config["n_binary_steps_to_reset_sdf"],
                    n_linearization_steps=config["sdf_reset_linearization_n_steps"],
                    enforce_std=config["sdf_reset_linearization_enforce_std"] if config["n_binary_steps_to_reset_sdf"] > 0 else None,
                )  # Between -1 and 1
                base_occupancy = convert_sdf_to_occupancy(base_occupancy)  # Between 0.005 and 0.995
                
                # Reshape base occupancy to make it (N_sampled_gaussians, 9)
                base_occupancy = unflatten_voronoi_features(
                    base_occupancy, 
                    n_voronoi_per_gaussians=9
                )  # (N_sampled_gaussians, 9)
                
                # Logic for resetting occupancy values
                if config["learnable_sdf_reset_mode"] == "ema":
                    print(f"[INFO] Resetting learnable SDF with EMA.")
                    _n_ema = (gaussians._base_occupancy[delaunay_xyz_idx] != 0.).sum().item()
                    _n_voronoi = base_occupancy.view(-1).shape[0]
                    print(f"          > Number of points to reset with EMA: {_n_ema}/{_n_voronoi}")
                    sdf_alpha_ema = config["learnable_sdf_reset_alpha_ema"]
                    new_occupancy =  torch.where(
                        gaussians._base_occupancy[delaunay_xyz_idx] != 0.,  # Points that have been sampled before
                        (sdf_alpha_ema * base_occupancy 
                            + (1. - sdf_alpha_ema) * gaussians.get_occupancy[delaunay_xyz_idx]),
                        base_occupancy,
                    ).clamp(min=0.005, max=0.995)
                    
                elif config["learnable_sdf_reset_mode"] == "none":
                    new_occupancy = None
                    
                else:
                    raise ValueError(f"Invalid learnable SDF reset mode: {config['learnable_sdf_reset_mode']}")
                
                # Reset occupancy values for the sampled gaussians
                gaussians.reset_occupancy(
                    base_occupancy=base_occupancy, 
                    occupancy=new_occupancy,
                    gaussian_idx=delaunay_xyz_idx, 
                )
                
                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()
            
                reset_sdf_values = False # Reset flag after computation
        
        # Convert learnable occupancy values to SDF
        if delaunay_xyz_idx is not None:
            current_occupancy = gaussians.get_occupancy[delaunay_xyz_idx]  # (N_sampled_gaussians, 9)
        else:
            current_occupancy = gaussians.get_occupancy  # (N_gaussians, 9)
        current_voronoi_sdf = convert_occupancy_to_sdf(
            flatten_voronoi_features(current_occupancy)
        )  # (N_voronoi_points, )

        # --- Marching Tetrahedra ---
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(
            vertices=voronoi_points[None],
            tets=delaunay_tets,
            sdf=current_voronoi_sdf.reshape(1, -1), # Use the computed SDF for this iteration
            scales=voronoi_scale[None]
        )
        end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
        end_scales = scale_list[0]  # (N_verts, 2, 1)
        
        norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)
        verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]        
        faces = faces_list[0]  # (N_faces, 3)

        # --- Filtering ---
        # Frustum filtering
        faces_mask = is_in_view_frustum(verts, viewpoint_cam)[faces].any(axis=1)
        
        # GOF filtering for large edges
        if config["filter_large_edges"] or config["collapse_large_edges"]:
            dmtet_distance = torch.norm(end_points[:, 0, :] - end_points[:, 1, :], dim=-1)
            dmtet_scale = end_scales[:, 0, 0] + end_scales[:, 1, 0]
            dmtet_vertex_mask = (dmtet_distance <= dmtet_scale)
            
        if config["filter_large_edges"]:
            dmtet_face_mask = dmtet_vertex_mask[faces].all(axis=1)
            faces_mask = faces_mask & dmtet_face_mask
            
        if config["collapse_large_edges"]:
            min_end_points = end_points[
                np.arange(end_points.shape[0]), 
                end_sdf.argmin(dim=1).flatten().cpu().numpy()
            ]  # TODO: Do the computation only for filtered vertices
            verts = torch.where(dmtet_vertex_mask[:, None], verts, min_end_points)

        # --- Build and Render Mesh ---
        mesh = Meshes(verts=verts, faces=faces[faces_mask])

        mesh_render_pkg = mesh_renderer(
            mesh,
            cam_idx=viewpoint_idx,
            return_depth=config["use_depth_loss"],
            return_normals=config["use_normal_loss"],
            use_antialiasing=True,
        )
        mesh_depth = (
            mesh_render_pkg["depth"].squeeze() 
            if config["use_depth_loss"] 
            else torch.zeros(viewpoint_cam.image_height, viewpoint_cam.image_width)
        )  # (H, W)
        mesh_normal_view = (
            mesh_render_pkg["normals"].squeeze() @ viewpoint_cam.world_view_transform[:3,:3] 
            if config["use_normal_loss"] 
            else torch.zeros(viewpoint_cam.image_height, viewpoint_cam.image_width, 3)
        )  # (H, W, 3)
        rasterization_mask = mesh_depth > 0.  # (H, W)
        
        # Reset occupancy labels
        if (
            config["use_occupancy_labels_loss"] 
            and (
                (iteration % config["reset_occupancy_labels_every"] == 0)  # Every N iterations
                or (iteration == config["start_iter"])  # First iteration
                or reset_occupancy_labels_for_new_delaunay_sites  # If not fixing sites and downsampling, compute labels for sampled sites
            )
        ):
            print(f"[INFO] Resetting occupancy labels at iteration {iteration}.")
            voronoi_occupancy_labels, _ = evaluate_mesh_occupancy(
                points=voronoi_points,
                views=scene.getTrainCameras().copy(),
                mesh=Meshes(verts=verts, faces=faces),
                masks=None,
                return_colors=True,
                use_scalable_renderer=config["use_scalable_renderer"],
            )
            print(f"[INFO] Points with label > 0.5: {torch.sum(voronoi_occupancy_labels > 0.5) / voronoi_occupancy_labels.numel()}")

        # --- Compute Losses ---
        mesh_depth_ratio = config["depth_ratio"]
        
        # Mesh Depth Loss
        if config["use_depth_loss"]:
            gaussians_depth = (
                (1. - mesh_depth_ratio) * render_pkg["expected_depth"] 
                + mesh_depth_ratio * render_pkg["median_depth"]
            ).squeeze()  # (H, W)
            
            if config["mesh_depth_loss_type"] == "log":
                mesh_depth_loss = torch.log(1. + (mesh_depth - gaussians_depth).abs() / gaussians.spatial_lr_scale)  # (H, W)

            elif config["mesh_depth_loss_type"] == "normal":
                mesh_depth_loss = depth_double_to_normal(
                    viewpoint_cam,
                    mesh_depth.squeeze()[None],
                    gaussians_depth.squeeze()[None],
                )  # (2, 3, H, W)
                mesh_depth_loss = 1. - (mesh_depth_loss[0] * mesh_depth_loss[1]).sum(dim=0)  # (H, W)

            else:
                raise ValueError(f"Invalid mesh depth loss type: {config['mesh_depth_loss_type']}")
            
            mesh_depth_loss = lambda_mesh_depth * (mesh_depth_loss * rasterization_mask).mean()
        else:
            mesh_depth_loss = torch.zeros(size=(), device=gaussians._xyz.device)

        # Mesh Normal Loss
        if config["use_normal_loss"]:
            if config["use_depth_normal"]:
                # Compute normals from Gaussian depth map
                depth_middepth_normal = depth_double_to_normal(
                    viewpoint_cam,
                    render_pkg["expected_depth"],
                    render_pkg["median_depth"]
                )
                gaussians_normal_view = (
                    (1. - mesh_depth_ratio) * depth_middepth_normal[0]
                    + mesh_depth_ratio * depth_middepth_normal[1]
                ).permute(1, 2, 0) # (H, W, 3)
            else:
                # Use rendered normals directly (already in view space)
                gaussians_normal_view = render_pkg["normal"].permute(1, 2, 0)  # (H, W, 3)

            # Compute cosine similarity loss (1 - |dot_product|).
            #
            # To do this, we flip the mesh normals to make the loss invariant to the sign of the mesh normal.
            # Indeed, we just want to make sure the planes of both the mesh and the gaussians are aligned,
            # so we don't really care about the direction of the normal.
            #
            # This might be needed in scenarios where the Delaunay triangulation is not updated for a while,
            # so that the mesh could self-intersect and have flipped normals.
            #
            # For computing the loss, we just need to use .abs() on the dot product.
            # We also explicitly flip the mesh normals for logging purposes.

            normal_dot_product = (mesh_normal_view * gaussians_normal_view).sum(dim=-1, keepdim=True)  # (H, W, 1)
            mesh_normal_loss = 1. - normal_dot_product.abs()  # (H, W, 1)
            mesh_normal_loss = lambda_mesh_normal * (mesh_normal_loss * rasterization_mask.unsqueeze(-1)).mean()
        else:
            mesh_normal_loss = torch.zeros(size=(), device=gaussians._xyz.device)
            
        # Enforce occupied centers
        if config["enforce_occupied_centers"]:
            # Get sdf values for centers of sampled Gaussians
            if mesh_state["surface_delaunay_xyz_idx"] is not None:
                gaussians_occupancy = gaussians.get_occupancy[mesh_state["surface_delaunay_xyz_idx"]]  # (N_surface_gaussians, 9)
                gaussians_occupancy = gaussians_occupancy[:, -1]  # (N_surface_gaussians, )
            else:
                gaussians_occupancy = current_occupancy[:, -1]
            occupied_centers_loss = config["occupied_centers_weight"] * (config["sdf_default_isosurface"] - gaussians_occupancy).clamp(min=0.).mean()
        else:
            occupied_centers_loss = torch.zeros(size=(), device=gaussians._xyz.device)
            
        # Occupancy labels loss
        if config["use_occupancy_labels_loss"]:
            occupancy_labels_loss = config["occupancy_labels_loss_weight"] * (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    flatten_voronoi_features(
                        gaussians.get_occupancy_logit if delaunay_xyz_idx is None
                        else gaussians.get_occupancy_logit[delaunay_xyz_idx]
                    ),
                    voronoi_occupancy_labels
                )
            ) * (voronoi_occupancy_labels > 0.5).float()
            occupancy_labels_loss = occupancy_labels_loss.mean()
        else:
            occupancy_labels_loss = torch.zeros(size=(), device=gaussians._xyz.device)

    # --- Return Results ---
    total_mesh_loss = (
        mesh_depth_loss 
        + mesh_normal_loss 
        + occupied_centers_loss 
        + occupancy_labels_loss
    )
    
    # --- Update State ---
    # Store updated filter parameters
    mesh_state["delaunay_xyz_idx"] = delaunay_xyz_idx
    # Store updated occupancy labels
    mesh_state["voronoi_occupancy_labels"] = voronoi_occupancy_labels
    # Store Updated Delaunay tetrahedra
    mesh_state["delaunay_tets"] = delaunay_tets
    # Reset flags were potentially set back to False inside the logic
    mesh_state["reset_delaunay_samples"] = reset_delaunay_samples
    mesh_state["reset_sdf_values"] = reset_sdf_values

    return {
        "mesh_loss": total_mesh_loss,
        "mesh_depth_loss": mesh_depth_loss.detach(),
        "mesh_normal_loss": mesh_normal_loss.detach(),
        "occupied_centers_loss": occupied_centers_loss.detach(),
        "occupancy_labels_loss": occupancy_labels_loss.detach(),
        "updated_state": mesh_state,
        "mesh_render_pkg": {
            "depth": mesh_depth,
            "normals": mesh_normal_view,
        }, # Contains depth/normals for logging
        "voronoi_points_count": voronoi_points_count,
    }


def reset_mesh_state_at_next_iteration(mesh_state):
    mesh_state["reset_delaunay_samples"] = True
    mesh_state["reset_sdf_values"] = True
    mesh_state["delaunay_tets"] = None
    return mesh_state
