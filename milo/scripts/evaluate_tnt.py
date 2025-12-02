import os
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


scenes_dict = {
    "Barn": {
        "imp_metric": "outdoor",
        "sampling_factor": None,  # Default sampling factor
        "decoupled_appearance": True,
    },
    "Caterpillar": {
        "imp_metric": "outdoor",
        "sampling_factor": None,  # Default sampling factor
        "decoupled_appearance": True,
    },
    "Courthouse": {
        "imp_metric": "outdoor",
        # Contrary to other scenes, Courthouse is a huge scene with 1k+ images. 
        # Thus, sampling more Gaussians is better: We increase the sampling factor to 1.0.
        "sampling_factor": 1.0,  
        # Since the scene is huge, we can also save some memory by not using decoupled appearance
        "decoupled_appearance": False, 
    },
    "Ignatius": {
        "imp_metric": "outdoor",
        "sampling_factor": None,  # Default sampling factor
        "decoupled_appearance": True,
    },
    "Meetingroom": {
        "imp_metric": "indoor",
        "sampling_factor": None,  # Default sampling factor
        "decoupled_appearance": True,
    },
    "Truck": {
        "imp_metric": "outdoor",
        "sampling_factor": None,  # Default sampling factor
        "decoupled_appearance": True,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--gt_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # Model and training parameters
    parser.add_argument("--rasterizer", type=str, default="radegs", choices=["radegs", "gof"])
    parser.add_argument("--dense_gaussians", action="store_true")
    parser.add_argument("--mesh_config", type=str, default="default")
    parser.add_argument("--no_mesh_regularization", action="store_true")
    
    # Depth
    parser.add_argument("--depth_order", action="store_true")
    parser.add_argument("--depth_order_config", type=str, default="default")

    # GPU handling
    parser.add_argument("--gpu_device", type=str, default="0")
    parser.add_argument("--data_on_gpu", action="store_true")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=None)

    args = parser.parse_args()
    
    for scene_name, scene_params in scenes_dict.items():
        print(f"\n[INFO] =====Evaluating {scene_name}=====")
        
        # Automatically set dense gaussians if the mesh config is highres or veryhighres
        use_dense_gaussians = args.dense_gaussians or (args.mesh_config in ["highres", "veryhighres"])
        
        # Set output name
        output_name = f"{scene_name}_{args.rasterizer}_{args.mesh_config}"
        if use_dense_gaussians:
            output_name += "_dense"
        if args.depth_order:
            output_name += f"_depthorder_{args.depth_order_config}"
        
        # Training command
        train_command = " ".join([
            f"CUDA_VISIBLE_DEVICES={args.gpu_device} python train.py",
            f"-s {os.path.join(args.data_dir, scene_name)}",
            f"-m {os.path.join(args.output_dir, output_name)}",
            f"--imp_metric {scene_params['imp_metric']}",
            f"--rasterizer {args.rasterizer}",
            f"--mesh_config {args.mesh_config}",
            f"--sampling_factor {scene_params['sampling_factor']}" if scene_params['sampling_factor'] is not None else "",
            "--dense_gaussians" if use_dense_gaussians else "",
            "--decoupled_appearance" if scene_params["decoupled_appearance"] else "",
            "--data_device cpu" if not args.data_on_gpu else "",
            "--eval",
            "--depth_order" if args.depth_order else "",
            f"--depth_order_config {args.depth_order_config}" if args.depth_order else "",
            f"--wandb_project {args.wandb_project}" if args.wandb_project is not None else "",
            f"--wandb_entity {args.wandb_entity}" if args.wandb_entity is not None else "",
            f"--log_interval {args.log_interval}" if args.log_interval is not None else "",
            "--no_mesh_regularization" if args.no_mesh_regularization else "",
        ])
        
        # Mesh extraction command
        mesh_command = " ".join([
            f"CUDA_VISIBLE_DEVICES={args.gpu_device} python mesh_extract_sdf.py",
            f"-s {os.path.join(args.data_dir, scene_name)}",
            f"-m {os.path.join(args.output_dir, output_name)}",
            f"--rasterizer {args.rasterizer}",
            f"--config {args.mesh_config}",
            "--data_device cpu" if not args.data_on_gpu else "",
            "--eval",
        ])
        
        # Evaluation command
        eval_command = " ".join([
            f"python eval/tnt/run.py",
            "--dataset-dir", os.path.join(
                args.gt_dir, scene_name
            ),
            "--traj-path", os.path.join(
                args.gt_dir, scene_name, f"{scene_name}_COLMAP_SfM.log"
            ),
            "--ply-path", os.path.join(
                args.output_dir, output_name, "mesh_learnable_sdf.ply"
            ),
        ])
        
        # Run commands
        print("\n[INFO] Running training command :", train_command, sep="\n")
        os.system(train_command)
        if not args.no_mesh_regularization:
            print("\n[INFO] Running mesh extraction command :", mesh_command, sep="\n")
            os.system(mesh_command)
            print("\n[INFO] Running evaluation command :", eval_command, sep="\n")
            os.system(eval_command)
