#adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
import numpy as np
import torch
from functools import partial
from scene import Scene
import os
from os import makedirs
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render_simp
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from scene.mesh import Meshes
from utils.tetmesh import marching_tetrahedra
from regularization.sdf.integration import evaluate_cull_sdf_values as compute_sdf_with_integration
from regularization.sdf.depth_fusion import evaluate_sdf_values as compute_sdf_with_depth_fusion
from regularization.sdf.depth_fusion import evaluate_mesh_colors_all_vertices
import time


@torch.no_grad()
def marching_tetrahedra_with_binary_search(
    model_path, iteration, views, 
    scene, gaussians: GaussianModel, 
    pipeline, background, kernel_size, 
    n_delaunay_gaussians=None, 
    mtet_on_cpu=False,
    sdf_mode="integration",
    n_binary_steps=8,
    isosurface_value=0.5,
    trunc_margin=None,
):    
    # Sample a subset of Gaussians for generating Gaussian pivots
    n_nonzero = (gaussians._base_occupancy != 0.).any(dim=-1).sum().item() if gaussians.learn_occupancy else 0
    
    #    > If no number of Gaussians is provided, reuse Gaussians used for training
    if n_nonzero > 0 and (n_delaunay_gaussians is None):
        delaunay_xyz_idx = (gaussians._base_occupancy != 0.).any(dim=-1).nonzero().squeeze()
        print(f"[INFO] Using the {delaunay_xyz_idx.shape[0]} Gaussians from training for generating pivots.")
    
    #    > Else, sample a novel subset of Gaussians
    else:
        n_gaussians_to_sample_from = gaussians._xyz.shape[0]
        n_max_gaussians_for_delaunay = n_gaussians_to_sample_from if n_delaunay_gaussians is None else n_delaunay_gaussians
        downsample_gaussians_for_delaunay = n_max_gaussians_for_delaunay < n_gaussians_to_sample_from
        if downsample_gaussians_for_delaunay:
            delaunay_xyz_idx = gaussians.sample_surface_gaussians(
                scene=scene,
                render_simp=render_simp,
                iteration=iteration,
                args=args,
                pipe=pipeline,
                background=background,
                n_samples=n_max_gaussians_for_delaunay,
            )
            print(f"[INFO] Downsampled {n_gaussians_to_sample_from} Gaussians to {delaunay_xyz_idx.shape[0]} Gaussians.")
        else:
            delaunay_xyz_idx = None
            print(f"[INFO] Using all {n_gaussians_to_sample_from} Gaussians for generating pivots.")
                
    # Generate Gaussian pivots
    points, points_scale = gaussians.get_tetra_points(xyz_idx=delaunay_xyz_idx)
    print(f"[INFO] Extracted {points.shape[0]} Delaunay sites from Gaussians.")
    t0 = time.time()
    
    # Compute Delaunay triangulation for the Gaussian pivots
    cells = cpp.triangulate(points)
    print(f"[INFO] Triangulated {points.shape[0]} Delaunay sites in {time.time() - t0} seconds.")
    
    # Compute SDF values for all Gaussian pivots
    mask = None
    if sdf_mode == "integration":
        sdf_function = partial(compute_sdf_with_integration, isosurface_value=isosurface_value, integrate_func=integrate)
    elif sdf_mode == "depth_fusion":
        sdf_function = partial(compute_sdf_with_depth_fusion, trunc_margin=trunc_margin, render_func=render)
    else:
        raise ValueError(f"Invalid sdf mode: {sdf_mode}")
    sdf = sdf_function(points, views, mask, gaussians, pipeline, background, kernel_size)
    torch.cuda.empty_cache()

    # Apply marching tetrahedra to get the mesh.
    # This requires a lot of memory, so you might want to move it to cpu.
    if mtet_on_cpu:
        print("[INFO] Running marching_tetrahedra on CPU.")
        verts_list, scale_list, faces_list, interp_v = marching_tetrahedra(points.cpu()[None], cells.cpu().long(), sdf[None].cpu(), points_scale[None].cpu())
    else:
        print("[INFO] Running marching_tetrahedra on GPU.")
        verts_list, scale_list, faces_list, interp_v = marching_tetrahedra(points[None], cells.cuda().long(), sdf[None], points_scale[None])
    del points
    del points_scale
    del cells
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    end_points, end_sdf, end_scales = end_points.cuda(), end_sdf.cuda(), end_scales.cuda()
    end_idx = interp_v[0].cuda()
    
    faces=faces_list[0]
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
    
    # Refine result of marching tetrahedra with binary search along intersected edges
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        
        mid_sdf = sdf_function(mid_points, views, mask, gaussians, pipeline, background, kernel_size)
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        points = (left_points + right_points) / 2

    # Compute vertex colors
    print("[INFO] Computing vertex colors...")
    point_colors = evaluate_mesh_colors_all_vertices(
        views=views, 
        mesh=Meshes(verts=points, faces=faces),
        masks=None,
        use_scalable_renderer=True,
    )
    point_colors=(point_colors.cpu().numpy() * 255).astype(np.uint8)
    
    # Create mesh
    points = points.cpu().numpy()
    faces = faces.cpu().numpy()
    mesh = trimesh.Trimesh(
        vertices=points, 
        faces=faces, 
        vertex_colors=point_colors, 
        process=False
    )
    
    # Filter mesh
    vertice_mask = (distance <= scale).cpu().numpy()
    face_mask = vertice_mask[faces].all(axis=1)
    mesh.update_vertices(vertice_mask)
    mesh.update_faces(face_mask)

    # Export mesh
    mesh.export(os.path.join(model_path,f"mesh_{sdf_mode}_sdf.ply"))

    
def extract_mesh(
    dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
    n_delaunay_sites=None, mtet_on_cpu=False, 
    sdf_mode="integration", n_binary_steps=8, 
    isosurface_value=0.5, trunc_margin=None,
):
    with torch.no_grad():
        # Load scene and Gaussian model
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        if gaussians.learn_occupancy:
            gaussians.set_occupancy_mode("occupancy_shift")
        
        print(f"[INFO] Loaded Gaussian Model from {os.path.join(dataset.model_path, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')}")
        print(f"[INFO]    > Number of Gaussians: {gaussians._xyz.shape[0]}")
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        try:
            kernel_size = dataset.kernel_size
        except:
            print("No kernel size found in dataset, using 0.0")
            kernel_size = 0.0
            
        if n_delaunay_sites < 0:
            n_delaunay_gaussians = None
        else:
            n_delaunay_gaussians = n_delaunay_sites // 9
        
        marching_tetrahedra_with_binary_search(
            model_path=dataset.model_path,
            iteration=iteration, 
            views=scene.getTrainCameras(), 
            scene=scene,
            gaussians=gaussians, 
            pipeline=pipeline, 
            background=background, 
            kernel_size=kernel_size, 
            n_delaunay_gaussians=n_delaunay_gaussians, 
            mtet_on_cpu=mtet_on_cpu,
            sdf_mode=sdf_mode,
            n_binary_steps=n_binary_steps,
            isosurface_value=isosurface_value,
            trunc_margin=trunc_margin,
        )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=18000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--rasterizer", default="gof", choices=["radegs", "gof"])
    parser.add_argument("--sdf_mode", default="integration", choices=["integration", "depth_fusion"])
    parser.add_argument("--mtet_on_cpu", action="store_true")
    # Integration
    parser.add_argument("--n_binary_steps", default=8, type=int)
    parser.add_argument("--isosurface_value", default=-1., type=float)
    # Depth fusion
    parser.add_argument("--trunc_margin", default=-1., type=float)
    # For sampling Gaussians for Delaunay triangulation
    parser.add_argument("--n_delaunay_sites", default=-1, type=int, 
                        help="Max number of pivots to use for Delaunay triangulation.")
    parser.add_argument("--imp_metric", default='none', type=str)
    parser.add_argument("--warn_until_iter", default=3000, type=int)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    print(f"[ INFO ] Using rasterizer: {args.rasterizer}")
    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
        from gaussian_renderer.radegs import integrate_radegs as integrate
    elif args.rasterizer == "gof":
        from gaussian_renderer.gof import render_gof as render
        from gaussian_renderer.gof import integrate_gof as integrate
    else:
        raise ValueError(f"Invalid rasterizer: {args.rasterizer}")
        
    if args.n_delaunay_sites > 0:
        if args.imp_metric == 'none':
            raise ValueError("imp_metric must be specified for using delaunay downsampling: Either 'indoor' or 'outdoor'")
    
    # For integration mode
    if args.isosurface_value < 0:
        args.isosurface_value = None
    if args.sdf_mode in ["integration"]:
        if args.isosurface_value is None:
            args.isosurface_value = 0.5
        print(f"[ INFO ] Using isosurface value {args.isosurface_value} for {args.sdf_mode}.")
    
    # For depth fusion mode
    if args.trunc_margin < 0:
        args.trunc_margin = None
    if args.sdf_mode in ["depth_fusion"]:
        if args.trunc_margin is None:
            print(f"[ INFO ] Using default truncation margin for {args.sdf_mode}.")
        
    extract_mesh(
        model.extract(args), 
        args.iteration, 
        pipeline.extract(args), 
        n_delaunay_sites=args.n_delaunay_sites, 
        mtet_on_cpu=args.mtet_on_cpu,
        sdf_mode=args.sdf_mode,
        n_binary_steps=args.n_binary_steps,
        isosurface_value=args.isosurface_value,
        trunc_margin=args.trunc_margin,
    )
    