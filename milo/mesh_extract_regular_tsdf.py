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
from scene import Scene
import os
from gaussian_renderer import render_radegs, render_gof
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.regular_tsdf_utils import GaussianExtractor, post_process_mesh

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--depth_ratio", default=0.6, type=float, help='Mesh: depth ratio for TSDF')
    parser.add_argument("--rasterizer", default="radegs", type=str, choices=["radegs", "gof"])
    
    # ---To change mesh size and resolution---
    parser.add_argument("--radius_factor", default=2.0, type=float, help='Mesh: radius factor for TSDF (in scene radius units)')
    parser.add_argument("--sdf_trunc_factor", default=5.0, type=float, help='Mesh: sdf truncation factor for TSDF (in voxel units)')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction (in voxels)')

    # ---Following args can be automatically computed from the above args---
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if args.rasterizer == "radegs":
        render = render_radegs
    elif args.rasterizer == "gof":
        render = render_gof
    else:
        raise ValueError(f"Invalid rasterizer: {args.rasterizer}")
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color, depth_ratio=args.depth_ratio)    

    print("export mesh ...")

    # set the active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(scene.getTrainCameras())

    # extract the mesh
    depth_trunc = (gaussExtractor.radius * args.radius_factor) if args.depth_trunc < 0  else args.depth_trunc
    voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
    sdf_trunc = args.sdf_trunc_factor * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
    
    print(f"\nRadius factor: {args.radius_factor}")
    print(f"Mesh resolution: {args.mesh_res}")
    print(f"Sdf truncation factor: {args.sdf_trunc_factor}")
    
    print(f"\nEstimated radius: {gaussExtractor.radius}")
    print(f"Depth truncation: {depth_trunc}")
    print(f"Voxel size: {voxel_size}")
    print(f"Sdf truncation: {sdf_trunc}\n")
    
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    
    # save the mesh
    name = f'mesh_regular_tsdf_res{args.mesh_res}.ply'
    mesh_path = os.path.join(args.model_path, name)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print("\nmesh saved at {}".format(mesh_path))
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of faces: {len(mesh.triangles)}")

    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    post_mesh_path = os.path.join(args.model_path, name.replace('.ply', '_post.ply'))
    o3d.io.write_triangle_mesh(post_mesh_path, mesh_post)
    print("\nmesh post processed saved at {}".format(post_mesh_path))
    print(f"Number of vertices: {len(mesh_post.vertices)}")
    print(f"Number of faces: {len(mesh_post.triangles)}")