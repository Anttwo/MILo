import os
import sys
import gc
import yaml
from functools import partial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')
sys.path.append(ROOT_DIR)
sys.path.append(SUBMODULES_DIR)
sys.path.append(os.path.join(SUBMODULES_DIR, 'Depth-Anything-V2'))

import torch
from random import randint
from utils.loss_utils import l1_loss, L1_loss_appearance
from fused_ssim import fused_ssim

from gaussian_renderer import network_gui
from gaussian_renderer import render_imp, render_simp, render_depth, render_full
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

import numpy as np
import time

from utils.geometry_utils import depth_to_normal
from utils.log_utils import log_training_progress
from regularization.regularizer.depth_order import (
    initialize_depth_order_supervision,
    compute_depth_order_regularization,
)
from regularization.regularizer.mesh import (
    initialize_mesh_regularization,
    compute_mesh_regularization,
    reset_mesh_state_at_next_iteration,
)

def training(
    dataset, opt, pipe, 
    testing_iterations, saving_iterations, 
    checkpoint_iterations, checkpoint, 
    debug_from, args, 
    depth_order_config, mesh_config,
    log_interval,
):
    # ---Prepare logger--- 
    run = prepare_output_and_logger(dataset, args)
    
    # ---Initialize scene and Gaussians---
    first_iter = 0
    use_mip_filter = not args.disable_mip_filter
    gaussians = GaussianModel(
        sh_degree=dataset.sh_degree, 
        use_mip_filter=use_mip_filter, 
        learn_occupancy=args.mesh_regularization,
        use_appearance_network=args.decoupled_appearance,
        use_radegs_densification=True,
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    print(f"[INFO] Using 3D Mip Filter: {gaussians.use_mip_filter}")
    print(f"[INFO] Using learnable SDF: {gaussians.learn_occupancy}")
    if args.dense_gaussians:
        print("[INFO] Using dense Gaussians.")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if args.mesh_regularization:
            if first_iter > mesh_config["start_iter"]:
                mesh_config["start_iter"] = first_iter + 1
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize culling stats
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    gaussians.init_culling(len(scene.getTrainCameras().copy()))
    
    # Initialize 3D Mip filter
    if use_mip_filter:
        gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

    # Additional variables
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    postfix_dict = {}
    ema_loss_for_log = 0.0
    ema_depth_normal_loss_for_log = 0.0
    
    # ---Prepare Mesh-In-the-Loop Regularization---
    if args.mesh_regularization:
        print("[INFO] Using mesh regularization.")
        mesh_renderer, mesh_state = initialize_mesh_regularization(
            scene=scene,
            config=mesh_config,
        )
    ema_mesh_depth_loss_for_log = 0.0
    ema_mesh_normal_loss_for_log = 0.0
    ema_occupied_centers_loss_for_log = 0.0
    ema_occupancy_labels_loss_for_log = 0.0
    
    # ---Prepare Depth-Order Regularization---    
    if args.depth_order:
        print("[INFO] Using depth order regularization.")
        print(f"        > Using expected depth with depth_ratio {depth_order_config['depth_ratio']} for depth order regularization.")
        depth_priors = initialize_depth_order_supervision(
            scene=scene,
            config=depth_order_config,
            device='cuda',
        )
    ema_depth_order_loss_for_log = 0.0
        
    # ---Log optimizable param groups---
    print(f"[INFO] Found {len(gaussians.optimizer.param_groups)} optimizable param groups:")
    n_total_params = 0
    for param in gaussians.optimizer.param_groups:
        name = param['name']
        n_params = len(param['params'])
        print(f"\n========== {name} ==========")
        print(f"Total number of param groups: {n_params}")
        for param_i in param['params']:
            print(f"   > Shape {param_i.shape}")
            n_total_params = n_total_params + param_i.numel()
    if gaussians.learn_occupancy:
        print(f"\n========== base_occupancy ==========")
        print(f"   > Not learnable")
        print(f"   > Shape {gaussians._base_occupancy.shape}")
    print(f"\nTotal number of optimizable parameters: {n_total_params}\n")
    
    # ---Start optimization loop---    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):   

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # ---Update SH degree---
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ---Select random viewpoint---
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_idx_stack = list(range(len(viewpoint_stack)))

        _random_view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
        viewpoint_cam = viewpoint_stack.pop(_random_view_idx)

        # ---Render scene---
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        reg_kick_on = iteration >= args.regularization_from_iter
        mesh_kick_on = args.mesh_regularization and (iteration >= mesh_config["start_iter"])
        depth_order_kick_on = args.depth_order
        
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background,
            require_coord=False, 
            require_depth=True and (reg_kick_on or mesh_kick_on or depth_order_kick_on),
        )

        # ---Compute losses---
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        gt_image = viewpoint_cam.original_image.cuda()

        # Rendering loss
        if args.decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # Depth-Normal Consistency Regularization
        if reg_kick_on:
            rendered_depth_to_normals: torch.Tensor = depth_to_normal(
                viewpoint_cam, 
                render_pkg["median_depth"],  # 1, H, W
                render_pkg["expected_depth"],  # 1, H, W
            )  # 3, H, W or 2, 3, H, W
            rendered_normals: torch.Tensor = render_pkg["normal"]  # 3, H, W
            
            if rendered_depth_to_normals.ndim == 4:
                # If shape is 2, 3, H, W
                reg_depth_ratio = 0.6
                normal_error_map = 1. - (rendered_normals[None] * rendered_depth_to_normals).sum(dim=1)  # 2, H, W
                depth_normal_loss = args.lambda_depth_normal * (
                    (1. - reg_depth_ratio) * normal_error_map[0].mean() 
                    + reg_depth_ratio * normal_error_map[1].mean()
                )
            else:
                # If shape is 3, H, W
                depth_normal_loss = args.lambda_depth_normal * (1 - (rendered_normals * rendered_depth_to_normals).sum(dim=0)).mean()
            
            loss = loss + depth_normal_loss
            
        # Depth Order Regularization
        # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
        # > In the paper, MILo does not rely on any learned prior. 
        if depth_order_kick_on:
            if depth_order_config["depth_ratio"] < 1.:
                depth_for_depth_order = (
                    (1. - depth_order_config["depth_ratio"]) * render_pkg["expected_depth"]
                    + depth_order_config["depth_ratio"] * render_pkg["median_depth"]
                )
            else:
                depth_for_depth_order = render_pkg["median_depth"]
                
            depth_prior_loss, _, do_supervision_depth, lambda_depth_order = compute_depth_order_regularization(
                iteration=iteration,
                rendered_depth=depth_for_depth_order,
                depth_priors=depth_priors,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                config=depth_order_config,
            )
                
            loss = loss + depth_prior_loss
            depth_order_kick_on = lambda_depth_order > 0
        
        # Mesh-In-the-Loop Regularization
        if mesh_kick_on:
            if args.detach_gaussian_rendering:
                detached_render_pkg = {
                    "render": render_pkg["render"].detach(),
                    "median_depth": render_pkg["median_depth"].detach(),
                    "expected_depth": render_pkg["expected_depth"].detach(),
                    "normal": render_pkg["normal"].detach(),
                }
            
            mesh_regularization_pkg = compute_mesh_regularization(
                iteration=iteration,
                render_pkg=detached_render_pkg if args.detach_gaussian_rendering else render_pkg,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                scene=scene,
                pipe=pipe,
                background=background,
                kernel_size=0.0,
                config=mesh_config,
                mesh_renderer=mesh_renderer,
                mesh_state=mesh_state,
                render_func=partial(render, require_coord=False, require_depth=True),
                weight_adjustment=100. / opt.iterations,
                args=args,
                integrate_func=integrate,
            )
            mesh_loss = mesh_regularization_pkg["mesh_loss"]
            mesh_depth_loss = mesh_regularization_pkg["mesh_depth_loss"]
            mesh_normal_loss = mesh_regularization_pkg["mesh_normal_loss"]
            occupied_centers_loss = mesh_regularization_pkg["occupied_centers_loss"]
            occupancy_labels_loss = mesh_regularization_pkg["occupancy_labels_loss"]
            mesh_state = mesh_regularization_pkg["updated_state"]
            mesh_render_pkg = mesh_regularization_pkg["mesh_render_pkg"]
            
            loss = loss + mesh_loss
        
        # ---Backward pass---
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # ---Logging---
            (
                postfix_dict,
                ema_loss_for_log, 
                ema_depth_normal_loss_for_log, 
                ema_mesh_depth_loss_for_log, ema_mesh_normal_loss_for_log, 
                ema_occupied_centers_loss_for_log, ema_occupancy_labels_loss_for_log, 
                ema_depth_order_loss_for_log
            ) = log_training_progress(
                args, iteration, log_interval, progress_bar, run,
                scene, gaussians, pipe, opt, background,
                viewpoint_idx, viewpoint_cam, render_pkg, 
                mesh_render_pkg if mesh_kick_on else None, 
                do_supervision_depth if depth_order_kick_on else None,
                reg_kick_on, mesh_kick_on, depth_order_kick_on,
                loss, depth_normal_loss if reg_kick_on else None, 
                mesh_depth_loss if mesh_kick_on else None, mesh_normal_loss if mesh_kick_on else None, 
                occupied_centers_loss if mesh_kick_on else None, occupancy_labels_loss if mesh_kick_on else None, 
                depth_prior_loss if depth_order_kick_on else None,
                mesh_config if mesh_kick_on else None, 
                postfix_dict, ema_loss_for_log, ema_depth_normal_loss_for_log, ema_mesh_depth_loss_for_log, 
                ema_mesh_normal_loss_for_log, ema_occupied_centers_loss_for_log, ema_occupancy_labels_loss_for_log,
                ema_depth_order_loss_for_log, testing_iterations, saving_iterations, render_imp,
            )

            # ---Densification---
            gaussians_have_changed = False
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats_radegs(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune_radegs(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    gaussians_have_changed = True
                    if not use_mip_filter:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # ---Reset mesh state if Gaussians have changed---
            if mesh_kick_on and gaussians_have_changed:
                mesh_state = reset_mesh_state_at_next_iteration(mesh_state)
                
            # ---Update 3D Mip Filter---
            if use_mip_filter and (
                (iteration == args.warn_until_iter)
                or (iteration % args.update_mip_filter_every == 0)
            ):
                if iteration < opt.iterations - args.update_mip_filter_every:
                    gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())
                else:
                    print(f"[INFO] Skipping 3D Mip Filter update at iteration {iteration}")

            # ---Optimizer step---
            if iteration < opt.iterations:
                if gaussians.use_appearance_network:
                    gaussians.optimizer.step()
                else:
                    visible = radii>0
                    gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none = True)

            # ---Save checkpoint---
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")  
                
        if iteration % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print('Num of Gaussians: %d'%(gaussians._xyz.shape[0]))
    
    if WANDB_FOUND:
        run.finish()
    
    return 


def prepare_output_and_logger(dataset, args):    
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create WandB run       
    global WANDB_FOUND
    WANDB_FOUND = (
        WANDB_FOUND
        and (args.wandb_project is not None)
        and (args.wandb_entity is not None)
    )
    if WANDB_FOUND:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
        )
    else:
        run=None
        print("[INFO] WandB not found, skipping logging.")
    return run


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # ----- Usual arguments -----
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=-1)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[19_999])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # ----- Rasterization technique -----
    parser.add_argument("--rasterizer", type=str, default="radegs", choices=["radegs", "gof"])
    
    # ----- Mesh-In-the-Loop Regularization -----
    parser.add_argument("--no_mesh_regularization", action="store_true")
    parser.add_argument("--mesh_config", type=str, default="default_regular_densification")
    # Gaussians management
    parser.add_argument("--dense_gaussians", action="store_true")
    parser.add_argument("--detach_gaussian_rendering", action="store_true")

    # ----- Simplification -----
    # > Inspired by Mini-Splatting2.
    # > Used Gaussian pivots selection.
    parser.add_argument("--imp_metric", required=True, type=str, choices=["outdoor", "indoor"])
    parser.add_argument("--warn_until_iter", type=int, default=3000)
    
    # ----- Depth-Normal consistency Regularization -----
    # > Inspired by 2DGS, GOF, RaDe-GS...
    parser.add_argument("--regularization_from_iter", type=int, default = 15_000)
    parser.add_argument("--lambda_depth_normal", type=float, default = 0.05)
    
    # ----- Depth Order Regularization (Learned Prior) -----
    # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
    # > In the paper, MILo does not rely on any learned prior.
    parser.add_argument("--depth_order", action="store_true")
    parser.add_argument("--depth_order_config", type=str, default="default_regular_densification")

    # ----- 3D Mip Filter -----
    # > Inspired by Mip-Splatting.
    parser.add_argument("--disable_mip_filter", action="store_true", default=False)
    parser.add_argument("--update_mip_filter_every", type=int, default=100)

    # ----- Appearance Network for Exposure-aware loss -----
    # > Inspired by GOF.
    parser.add_argument("--decoupled_appearance", action="store_true")

    # ----- Logging -----
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    if not -1 in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    args.mesh_regularization = not args.no_mesh_regularization
    
    if args.port == -1:
        args.port = np.random.randint(5000, 9000)
        print(f"Using random port: {args.port}")
    
    # Load depth order regularization config (not used in MILo paper)
    if args.depth_order:
        # Get depth order config file
        depth_order_config_file = os.path.join(BASE_DIR, "configs", "depth_order", f"{args.depth_order_config}.yaml")
        with open(depth_order_config_file, "r") as f:
            depth_order_config = yaml.safe_load(f)
    else:
        depth_order_config = None
        
    # Load mesh-in-the-loop regularization config
    if args.mesh_regularization:
        # Get mesh regularization config file
        mesh_config_file = os.path.join(BASE_DIR, "configs", "mesh", f"{args.mesh_config}.yaml")
        with open(mesh_config_file, "r") as f:
            mesh_config = yaml.safe_load(f)
        print(f"[INFO] Using mesh regularization with config: {args.mesh_config}")
    else:
        mesh_config = None
    
    # Message for imp_metric
    print(f"[INFO] Using importance metric: {args.imp_metric}.")
    
    # Message for detach_gaussian_rendering
    if args.detach_gaussian_rendering:
        print(f"[INFO] Detaching Gaussian rendering for mesh regularization.")
    
    # Import rendering function
    print(f"[INFO] Using {args.rasterizer} as rasterizer.")
    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
        from gaussian_renderer.radegs import integrate_radegs as integrate
    elif args.rasterizer == "gof":
        from gaussian_renderer.gof import render_gof as render
        from gaussian_renderer.gof import integrate_gof as integrate
        
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.cuda.synchronize()
    time_start=time.time()
    
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args,
        depth_order_config,
        mesh_config,
        args.log_interval,
    )

    torch.cuda.synchronize()
    time_end=time.time()
    time_total=time_end-time_start
    print('time: %fs'%(time_total))

    time_txt_path=os.path.join(args.model_path, r'time.txt')
    with open(time_txt_path, 'w') as f:  
        f.write(str(time_total)) 

    # All done
    print("\nTraining complete.")
