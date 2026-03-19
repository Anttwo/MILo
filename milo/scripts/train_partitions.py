import argparse
import os
import subprocess
import sys
import time
from math import ceil
from pathlib import Path
from typing import Iterable, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent


def parse_gpu_ids(raw_gpu_ids: Iterable[str]) -> List[str]:
    gpu_ids: List[str] = []
    for raw_value in raw_gpu_ids:
        for token in str(raw_value).split(','):
            token = token.strip()
            if token:
                gpu_ids.append(token)
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided.")
    return gpu_ids


def load_partition_image_counts(stats_path: Optional[Path]) -> dict:
    """Load {partition_name: num_images} from a text stats file."""
    counts = {}
    if stats_path is None or not stats_path.exists():
        return counts

    with stats_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            partition_name = parts[0]
            try:
                counts[partition_name] = int(parts[1])
            except ValueError:
                continue

    print(f"[INFO] Loaded image counts for {len(counts)} partitions from {stats_path}")
    return counts


def choose_sampling_factor(num_images: Optional[int], default_sampling_factor: Optional[float]) -> Optional[float]:
    """Optionally adapt MILo's sampling factor to partition size."""
    if default_sampling_factor is not None:
        return default_sampling_factor
    if num_images is None:
        return None
    if num_images < 200:
        return 0.4
    if num_images <= 600:
        return 0.6
    return 1.0


def list_partition_dirs(root_dir: Path) -> List[Path]:
    return sorted([path for path in root_dir.iterdir() if path.is_dir()])


def build_train_command(
    python_bin: str,
    train_script: Path,
    partition_dir: Path,
    output_dir: Path,
    port: int,
    args: argparse.Namespace,
    sampling_factor: Optional[float],
) -> List[str]:
    command = [
        python_bin,
        str(train_script),
        "-s",
        str(partition_dir),
        "-m",
        str(output_dir),
        "--imp_metric",
        args.imp_metric,
        "--rasterizer",
        args.rasterizer,
        "--mesh_config",
        args.mesh_config,
        "--port",
        str(port),
    ]

    if args.data_device:
        command.extend(["--data_device", args.data_device])
    if args.config_path:
        command.extend(["--config_path", args.config_path])
    if sampling_factor is not None:
        command.extend(["--sampling_factor", str(sampling_factor)])
    if args.log_interval is not None:
        command.extend(["--log_interval", str(args.log_interval)])
    if args.wandb_project:
        command.extend(["--wandb_project", args.wandb_project])
    if args.wandb_entity:
        command.extend(["--wandb_entity", args.wandb_entity])
    if args.iterations is not None:
        command.extend(["--iterations", str(args.iterations)])

    for flag_name in [
        "dense_gaussians",
        "decoupled_appearance",
        "depth_order",
        "no_mesh_regularization",
        "disable_mip_filter",
        "quiet",
        "eval",
    ]:
        if getattr(args, flag_name):
            command.append(f"--{flag_name}")

    if args.depth_order:
        command.extend(["--depth_order_config", args.depth_order_config])

    if args.extra_train_args:
        command.extend(args.extra_train_args)

    return command


def run_partition_batches(args: argparse.Namespace) -> None:
    root_dir = Path(args.root_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_script = (BASE_DIR / "train.py").resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Could not find train.py at {train_script}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    partitions = list_partition_dirs(root_dir)
    if not partitions:
        raise ValueError(f"No partition subdirectories found in {root_dir}")

    partition_counts = load_partition_image_counts(
        Path(args.partition_stats_path).resolve() if args.partition_stats_path else None
    )

    batch_size = len(gpu_ids)
    num_batches = ceil(len(partitions) / batch_size)

    print(f"[INFO] Found {len(partitions)} partitions under {root_dir}")
    print(f"[INFO] Using GPUs: {', '.join(gpu_ids)}")
    print(f"[INFO] Output root: {output_root}")

    for batch_idx in range(num_batches):
        batch_partitions = partitions[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        processes = []

        print(f"\n[INFO] Starting batch {batch_idx + 1}/{num_batches}")
        for slot_idx, partition_dir in enumerate(batch_partitions):
            gpu_id = gpu_ids[slot_idx]
            partition_name = partition_dir.name
            partition_output_dir = output_root / partition_name
            partition_output_dir.mkdir(parents=True, exist_ok=True)

            num_images = partition_counts.get(partition_name)
            sampling_factor = choose_sampling_factor(num_images, args.sampling_factor)
            port = args.base_port + batch_idx * batch_size + slot_idx

            if num_images is None:
                print(
                    f"[INFO] {partition_name}: image count unavailable, "
                    f"sampling_factor={sampling_factor if sampling_factor is not None else 'train.py default'}"
                )
            else:
                print(
                    f"[INFO] {partition_name}: num_images={num_images}, "
                    f"sampling_factor={sampling_factor if sampling_factor is not None else 'train.py default'}"
                )

            command = build_train_command(
                python_bin=args.python_bin,
                train_script=train_script,
                partition_dir=partition_dir,
                output_dir=partition_output_dir,
                port=port,
                args=args,
                sampling_factor=sampling_factor,
            )

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONUNBUFFERED"] = "1"

            print(f"[INFO] Launching on GPU {gpu_id}: {' '.join(command)}")
            process = subprocess.Popen(command, cwd=str(BASE_DIR), env=env)
            processes.append((partition_name, gpu_id, process))

            if args.launch_delay > 0:
                time.sleep(args.launch_delay)

        failed = []
        for partition_name, gpu_id, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failed.append((partition_name, gpu_id, return_code))

        if failed:
            print("[ERROR] The following partition jobs failed:")
            for partition_name, gpu_id, return_code in failed:
                print(f"    - {partition_name} on GPU {gpu_id}: return code {return_code}")
            raise RuntimeError(f"Batch {batch_idx + 1} failed for {len(failed)} partition(s).")

        print(f"[INFO] Batch {batch_idx + 1}/{num_batches} completed successfully")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train large-scene partitions with MILo by dispatching partitions across multiple GPUs."
    )
    parser.add_argument("root_dir", help="Directory containing partition subdirectories.")
    parser.add_argument("output_dir", help="Directory where per-partition outputs will be written.")
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        default=["0"],
        help="GPU ids to use, e.g. --gpu_ids 0 1 2 3 or --gpu_ids 0,1,2,3",
    )
    parser.add_argument("--imp_metric", required=True, choices=["indoor", "outdoor"])
    parser.add_argument("--rasterizer", default="radegs", choices=["radegs", "gof"])
    parser.add_argument("--mesh_config", default="default")
    parser.add_argument("--config_path", default="./configs/fast")
    parser.add_argument("--data_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sampling_factor", type=float, default=None)
    parser.add_argument("--partition_stats_path", default=None)
    parser.add_argument("--base_port", type=int, default=8000)
    parser.add_argument("--launch_delay", type=float, default=0.0)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--depth_order_config", default="default")
    parser.add_argument("--dense_gaussians", action="store_true")
    parser.add_argument("--decoupled_appearance", action="store_true")
    parser.add_argument("--depth_order", action="store_true")
    parser.add_argument("--no_mesh_regularization", action="store_true")
    parser.add_argument("--disable_mip_filter", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "extra_train_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to milo/train.py. Prefix them with '--'.",
    )
    return parser


if __name__ == "__main__":
    run_partition_batches(make_parser().parse_args())
