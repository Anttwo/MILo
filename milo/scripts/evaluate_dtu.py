import os
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


scene_names = [
    "scan24",
    "scan37",
    "scan40",
    "scan55",
    "scan63",
    "scan65",
    "scan69",
    "scan83",
    "scan97",
    "scan105",
    "scan106",
    "scan110",
    "scan114",
    "scan118",
    "scan122",
]

imp_metric = "indoor"
decoupled_appearance = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--gt_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # Model and training parameters
    parser.add_argument("--rasterizer", type=str, default="radegs", choices=["radegs", "gof"])
    parser.add_argument("--dense_gaussians", action="store_true")
    parser.add_argument("--mesh_config", type=str, default="default_dtu")
    
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
    
    for scene_name in scene_names:
        scan_id = scene_name[4:]
        print(f"\n[INFO] =====Evaluating Scan {scan_id}=====")
        
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
            f"CUDA_VISIBLE_DEVICES={args.gpu_device} python train_regular_densification.py",
            f"-s {os.path.join(args.data_dir, scene_name)}",
            f"-m {os.path.join(args.output_dir, output_name)}",
            f"-r 2",
            f"--imp_metric {imp_metric}",
            f"--rasterizer {args.rasterizer}",
            f"--mesh_config {args.mesh_config}",
            "--dense_gaussians" if use_dense_gaussians else "",
            "--decoupled_appearance" if decoupled_appearance else "",
            "--data_device cpu" if not args.data_on_gpu else "",
            "--depth_order" if args.depth_order else "",
            f"--depth_order_config {args.depth_order_config}" if args.depth_order else "",
            f"--wandb_project {args.wandb_project}" if args.wandb_project is not None else "",
            f"--wandb_entity {args.wandb_entity}" if args.wandb_entity is not None else "",
            f"--log_interval {args.log_interval}" if args.log_interval is not None else "",
        ])
        
        # Mesh extraction command
        mesh_command = " ".join([
            f"CUDA_VISIBLE_DEVICES={args.gpu_device} python ./eval/dtu/mesh_extract_dtu.py",
            f"-s {os.path.join(args.data_dir, scene_name)}",
            f"-m {os.path.join(args.output_dir, output_name)}",
            f"-r 2",
            f"--rasterizer {args.rasterizer}",
            "--data_device cpu" if not args.data_on_gpu else "",
        ])
        
        # Evaluation command
        eval_command = " ".join([
            f"python ./eval/dtu/evaluate_dtu_mesh.py", 
            f"-s {os.path.join(args.data_dir, scene_name)}",
            f"-m {os.path.join(args.output_dir, output_name)}",
            f"-r 2",
            f"--DTU {args.gt_dir}",
            f"--scan_id {scan_id}",
        ])
        
        # Run commands
        print("\n[INFO] Running training command :", train_command, sep="\n")
        os.system(train_command)
        print("\n[INFO] Running mesh extraction command :", mesh_command, sep="\n")
        os.system(mesh_command)
        print("\n[INFO] Running evaluation command :", eval_command, sep="\n")
        os.system(eval_command)
