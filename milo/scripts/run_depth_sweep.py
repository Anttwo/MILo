#!/usr/bin/env python3
"""
Run a small sweep of depth-only training configurations.

Each configuration executes `milo/depth_train.py` with 1000 iterations by default,
optionally restricting training to a single camera for reproducibility. After the
runs finish, a compact summary of the final metrics is emitted to stdout and
saved under `output/depth_sweep_summary.txt`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_CONFIGS: List[Dict] = [
    {
        "name": "lr0.7_clip7_dense",
        "initial_lr_scale": 0.70,
        "depth_clip_min": 0.1,
        "depth_clip_max": 7.0,
        "enable_densification": True,
    },
    {
        "name": "lr0.8_clip7_dense",
        "initial_lr_scale": 0.80,
        "depth_clip_min": 0.2,
        "depth_clip_max": 7.0,
        "enable_densification": True,
    },
    {
        "name": "lr0.9_clip6p5_dense",
        "initial_lr_scale": 0.90,
        "depth_clip_min": 0.3,
        "depth_clip_max": 6.5,
        "enable_densification": True,
    },
    {
        "name": "lr1.0_clip6_dense",
        "initial_lr_scale": 1.00,
        "depth_clip_min": 0.3,
        "depth_clip_max": 6.0,
        "enable_densification": True,
    },
    {
        "name": "lr1.15_clip6_dense",
        "initial_lr_scale": 1.15,
        "depth_clip_min": 0.4,
        "depth_clip_max": 6.0,
        "enable_densification": True,
    },
    {
        "name": "lr1.3_clip5p5_dense",
        "initial_lr_scale": 1.30,
        "depth_clip_min": 0.5,
        "depth_clip_max": 5.5,
        "enable_densification": True,
    },
    {
        "name": "lr0.75_clip7_no_dense",
        "initial_lr_scale": 0.75,
        "depth_clip_min": 0.1,
        "depth_clip_max": 7.0,
        "enable_densification": False,
    },
    {
        "name": "lr1.0_clip6_no_dense",
        "initial_lr_scale": 1.0,
        "depth_clip_min": 0.3,
        "depth_clip_max": 6.0,
        "enable_densification": False,
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep depth training configurations.")
    parser.add_argument("--ply_path", required=True, type=Path, help="Input Gaussian PLY.")
    parser.add_argument("--camera_poses", required=True, type=Path, help="Camera JSON file.")
    parser.add_argument("--depth_dir", required=True, type=Path, help="Directory of depth .npy files.")
    parser.add_argument("--output_root", type=Path, default=Path("runs/depth_sweep"), help="Base directory for sweep outputs.")
    parser.add_argument("--iterations", type=int, default=1000, help="Iterations per configuration.")
    parser.add_argument("--fixed_view_idx", type=int, default=0, help="Camera index to lock during sweep (-1 = random shuffling).")
    parser.add_argument("--cuda_blocking", action="store_true", help="Set CUDA_LAUNCH_BLOCKING=1 for each run.")
    parser.add_argument("--extra_arg", action="append", default=[], help="Extra CLI arguments passed verbatim to depth_train.py.")
    parser.add_argument("--resume_if_exists", action="store_true", help="Skip configs whose output directory already exists.")
    return parser


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_depth_train(
    script_path: Path,
    cfg: Dict,
    args: argparse.Namespace,
    run_dir: Path,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(script_path),
        "--ply_path",
        str(args.ply_path),
        "--camera_poses",
        str(args.camera_poses),
        "--depth_dir",
        str(args.depth_dir),
        "--output_dir",
        str(run_dir),
        "--iterations",
        str(args.iterations),
        "--initial_lr_scale",
        str(cfg["initial_lr_scale"]),
        "--log_depth_stats",
    ]
    if cfg.get("depth_clip_min", 0.0) > 0.0:
        cmd.extend(["--depth_clip_min", str(cfg["depth_clip_min"])])
    if cfg.get("depth_clip_max") is not None:
        cmd.extend(["--depth_clip_max", str(cfg["depth_clip_max"])])
    if cfg.get("enable_densification", False):
        cmd.append("--enable_densification")
    if args.fixed_view_idx >= 0:
        cmd.extend(["--fixed_view_idx", str(args.fixed_view_idx)])
    for extra in args.extra_arg:
        cmd.append(extra)

    env = os.environ.copy()
    if args.cuda_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    print(f"[SWEEP] Running {cfg['name']} -> {run_dir}")
    print("        Command:", " ".join(cmd))
    sys.stdout.flush()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[SWEEP] {cfg['name']} exited with code {result.returncode}")
    return result.returncode


def read_final_metrics(run_dir: Path) -> Optional[Dict]:
    log_path = run_dir / "logs" / "losses.jsonl"
    if not log_path.exists():
        return None
    last_line: Optional[str] = None
    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            last_line = line.strip()
    if not last_line:
        return None
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError:
        return None
    return {
        "iteration": data.get("iteration"),
        "depth_loss": data.get("depth_loss"),
        "pred_depth_mean": data.get("pred_depth_mean"),
        "target_depth_mean": data.get("target_depth_mean"),
        "pred_depth_max": data.get("pred_depth_max"),
        "pred_depth_min": data.get("pred_depth_min"),
        "target_depth_max": data.get("target_depth_max"),
        "target_depth_min": data.get("target_depth_min"),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    script_path = Path("milo/depth_train.py").resolve()
    ensure_directory(args.output_root)
    ensure_directory(Path("output"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_lines: List[str] = []

    for cfg in DEFAULT_CONFIGS:
        run_dir = args.output_root / f"{timestamp}_{cfg['name']}"
        if run_dir.exists() and args.resume_if_exists:
            print(f"[SWEEP] Skipping {cfg['name']} (directory exists).")
            metrics = read_final_metrics(run_dir)
        else:
            ensure_directory(run_dir)
            exit_code = run_depth_train(script_path, cfg, args, run_dir)
            if exit_code != 0:
                summary_lines.append(f"{cfg['name']}: FAILED (code {exit_code})")
                continue
            metrics = read_final_metrics(run_dir)

        if not metrics:
            summary_lines.append(f"{cfg['name']}: missing/invalid log")
            continue

        summary_lines.append(
            "{name}: depth_loss={loss:.4f} pred_mean={p_mean:.4f} target_mean={t_mean:.4f}".format(
                name=cfg["name"],
                loss=metrics.get("depth_loss", float("nan")),
                p_mean=metrics.get("pred_depth_mean", float("nan")),
                t_mean=metrics.get("target_depth_mean", float("nan")),
            )
        )

    summary_path = Path("output") / "depth_sweep_summary.txt"
    with open(summary_path, "a", encoding="utf-8") as summary_file:
        summary_file.write(f"# Sweep {timestamp}\n")
        for line in summary_lines:
            summary_file.write(line + "\n")
        summary_file.write("\n")

    print("\n[SWEEP] Summary:")
    for line in summary_lines:
        print("  -", line)
    print(f"[SWEEP] Full summary appended to {summary_path}")


if __name__ == "__main__":
    main()
