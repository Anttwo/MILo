#!/usr/bin/env python3
"""
Convert DISCOVERSE simulator exports (poses + RGB) into a MILo-compatible
COLMAP text scene.
"""

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import struct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DISCOVERSE simulator exports (poses + stereo RGB) into a MILo-compatible COLMAP text scene."
    )
    parser.add_argument("--source", required=True, help="Root directory containing DISCOVERSE exports.")
    parser.add_argument("--output", required=True, help="Destination directory for the COLMAP layout.")
    parser.add_argument("--poses-root", default=".", help="Relative subdirectory under source with pose JSON files.")
    parser.add_argument("--poses-glob", default="camera_poses_cam*.json",
                        help="Glob pattern (relative to poses-root) to pick pose JSON files.")
    parser.add_argument("--image-root", default=".", help="Relative subdirectory under source holding RGB images.")
    parser.add_argument("--image-pattern", default="rgb_img_{cam}_{frame_padded}.png",
                        help="Format string used to resolve image filenames. Keys: cam, frame, frame_padded, name, name_safe.")
    parser.add_argument("--frame-padding", type=int, default=6,
                        help="Zero padding width when building frame_padded (default: 6).")
    parser.add_argument("--orientation", choices=["camera_to_world", "world_to_camera"], default="camera_to_world",
                        help="Whether quaternions encode camera-to-world (default) or world-to-camera rotations.")
    parser.add_argument("--quaternion-order", choices=["wxyz", "xyzw"], default="wxyz",
                        help="Component order inside the pose JSON (default assumes [w,x,y,z]).")
    parser.add_argument("--camera-ids", type=int, nargs="*",
                        help="Optional explicit camera ids (for image naming) ordered like pose files.")
    parser.add_argument("--width", type=int, required=True, help="Default image width (pixels).")
    parser.add_argument("--height", type=int, required=True, help="Default image height (pixels).")
    parser.add_argument("--fx", type=float, required=True, help="Default focal length fx (pixels).")
    parser.add_argument("--fy", type=float, required=True, help="Default focal length fy (pixels).")
    parser.add_argument("--cx", type=float, required=True, help="Default principal point x.")
    parser.add_argument("--cy", type=float, required=True, help="Default principal point y.")
    parser.add_argument("--intrinsics-json", type=str,
                        help="Optional JSON file overriding intrinsics per camera id.")
    parser.add_argument("--copy-images", dest="copy_images", action="store_true",
                        help="Copy RGBs into output/images (default).")
    parser.add_argument("--link-images", dest="copy_images", action="store_false",
                        help="Use symlinks instead of copying RGBs.")
    parser.set_defaults(copy_images=True)
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing output files.")
    return parser.parse_args()


def load_pose_list(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict):
        for key in ("frames", "poses", "camera_poses"):
            if key in data and isinstance(data[key], list):
                frames = data[key]
                break
        else:
            raise ValueError(f"Unsupported JSON structure in {json_path}")
    else:
        raise ValueError(f"Unsupported JSON structure in {json_path}")

    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Frame {idx} in {json_path} is not an object")
        if "position" not in frame:
            raise ValueError(f"Frame {idx} in {json_path} misses 'position'")
        if "quaternion" not in frame:
            raise ValueError(f"Frame {idx} in {json_path} misses 'quaternion'")
    return frames


def normalize_quaternion(q: Iterable[float]) -> np.ndarray:
    q_arr = np.asarray(list(q), dtype=np.float64)
    norm = np.linalg.norm(q_arr)
    if norm == 0:
        raise ValueError("Encountered zero-length quaternion")
    return q_arr / norm


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0:
        q *= -1.0
    return q / np.linalg.norm(q)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", name)


def extract_frame_index(frame: dict) -> Optional[int]:
    for key in ("frame_id", "frame_index", "index", "idx"):
        if key in frame:
            try:
                return int(frame[key])
            except (TypeError, ValueError):
                pass
    name = frame.get("name") or frame.get("filename") or frame.get("image")
    if isinstance(name, str):
        matches = re.findall(r"traj[_-]?(\d+)", name)
        if matches:
            return int(matches[0])
        digits = re.findall(r"\d+", name)
        if digits:
            return int(digits[0])
    return None


def infer_camera_token(pose_path: Path, frames: List[dict]) -> Optional[int]:
    name_digits: List[int] = []
    for frame in frames:
        name = frame.get("name")
        if not isinstance(name, str):
            continue
        match = re.search(r"cam[_-]?(\d+)", name)
        if match:
            name_digits.append(int(match.group(1)))
    if name_digits:
        return int(np.median(name_digits))

    match = re.search(r"cam[_-]?(\d+)", pose_path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", pose_path.stem)
    if match:
        return int(match.group(1))
    return None


def build_image_candidates(args: argparse.Namespace, cam_token: int, frame_idx: int, frame_name: Optional[str]) -> List[str]:
    safe_name = sanitize_name(frame_name) if isinstance(frame_name, str) else None
    padded = f"{frame_idx:0{args.frame_padding}d}" if args.frame_padding and frame_idx is not None else str(frame_idx)
    fmt_values = {
        "cam": cam_token,
        "frame": frame_idx,
        "frame_padded": padded,
        "name": frame_name if isinstance(frame_name, str) else f"frame_{frame_idx}",
        "name_safe": safe_name if safe_name else f"frame_{frame_idx}",
    }
    candidates: List[str] = []
    try:
        candidates.append(args.image_pattern.format(**fmt_values))
    except KeyError:
        pass
    if frame_idx is not None:
        for pad in (6, 5, 4, 3, 2, 1, 0):
            if pad == args.frame_padding:
                continue
            fmt_values_mod = dict(fmt_values)
            if pad > 0:
                fmt_values_mod["frame_padded"] = f"{frame_idx:0{pad}d}"
            else:
                fmt_values_mod["frame_padded"] = str(frame_idx)
            try:
                candidate = args.image_pattern.format(**fmt_values_mod)
            except KeyError:
                continue
            if candidate not in candidates:
                candidates.append(candidate)
        if safe_name:
            fallback = [
                f"rgb_img_{cam_token}_{safe_name}.png",
                f"rgb_img_{cam_token}_{frame_idx}.png",
                f"rgb_img_{cam_token}_{frame_idx:06d}.png",
                f"{safe_name}.png",
            ]
            for cand in fallback:
                if cand not in candidates:
                    candidates.append(cand)
    return candidates


def resolve_image_path(images_root: Path, candidates: List[str]) -> Optional[Path]:
    for rel in candidates:
        candidate_path = images_root / rel
        if candidate_path.exists():
            return candidate_path
    return None


def load_intrinsics(args: argparse.Namespace) -> Dict[int, Dict[str, float]]:
    per_camera: Dict[int, Dict[str, float]] = {}
    if args.intrinsics_json:
        with open(args.intrinsics_json, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError("intrinsics JSON must map camera ids to parameter dicts")
        for key, entry in raw.items():
            try:
                cam_id = int(key)
            except ValueError as exc:
                raise ValueError(f"Invalid camera id '{key}' in intrinsics JSON") from exc
            required = {"width", "height", "fx", "fy", "cx", "cy"}
            if not required.issubset(entry.keys()):
                missing = required - set(entry.keys())
                raise ValueError(f"Intrinsics entry for camera {cam_id} missing {missing}")
            per_camera[cam_id] = {k: float(entry[k]) for k in required}
            per_camera[cam_id]["width"] = int(per_camera[cam_id]["width"])
            per_camera[cam_id]["height"] = int(per_camera[cam_id]["height"])
    return per_camera


def main():
    args = parse_args()

    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()
    pose_root = (source_root / args.poses_root).resolve()
    images_root = (source_root / args.image_root).resolve()

    intrinsics_map = load_intrinsics(args)

    pose_files = sorted(pose_root.glob(args.poses_glob))
    if not pose_files:
        raise FileNotFoundError(f"No pose files matched '{args.poses_glob}' under {pose_root}")

    if args.camera_ids and len(args.camera_ids) != len(pose_files):
        raise ValueError("Number of --camera-ids entries must match pose files found")

    default_intr = {"width": args.width, "height": args.height, "fx": args.fx,
                    "fy": args.fy, "cx": args.cx, "cy": args.cy}

    entries = []
    camera_id_map: Dict[int, int] = {}
    next_camera_colmap_id = 1
    next_image_id = 1

    for pose_idx, pose_path in enumerate(pose_files):
        frames = load_pose_list(pose_path)
        file_cam_token = infer_camera_token(pose_path, frames)
        explicit_cam_token = None
        if args.camera_ids:
            explicit_cam_token = args.camera_ids[pose_idx]
        cam_token_candidates = []
        if explicit_cam_token is not None:
            cam_token_candidates.append(explicit_cam_token)
        if file_cam_token is not None:
            cam_token_candidates.extend([file_cam_token, file_cam_token - 1, file_cam_token + 1])
        cam_token_candidates.extend([pose_idx, pose_idx + 1])
        candidate_list = [c for c in cam_token_candidates if c is not None and c >= 0]
        dedup_candidates = []
        for cand in candidate_list:
            if cand not in dedup_candidates:
                dedup_candidates.append(cand)
        if not dedup_candidates:
            dedup_candidates = [pose_idx]

        chosen_cam_token: Optional[int] = None
        first_frame = frames[0]
        frame_idx = extract_frame_index(first_frame)
        for cam_candidate in dedup_candidates:
            candidates = build_image_candidates(args, cam_candidate, frame_idx or 0, first_frame.get("name"))
            found = resolve_image_path(images_root, candidates)
            if found:
                chosen_cam_token = cam_candidate
                break
        if chosen_cam_token is None:
            raise FileNotFoundError(
                f"Unable to locate RGB for first frame in {pose_path.name}. Tried camera ids {dedup_candidates} under {images_root}"
            )

        if chosen_cam_token not in camera_id_map:
            camera_id_map[chosen_cam_token] = next_camera_colmap_id
            next_camera_colmap_id += 1

        intr = intrinsics_map.get(chosen_cam_token, default_intr)
        for key in ("width", "height", "fx", "fy", "cx", "cy"):
            if key not in intr:
                raise ValueError(f"Missing intrinsic '{key}' for camera {chosen_cam_token}")

        for f_idx, frame in enumerate(frames):
            position = np.asarray(frame["position"], dtype=np.float64).reshape(-1)
            if position.size != 3:
                raise ValueError(f"Position for frame {f_idx} in {pose_path.name} is not length 3")
            quaternion_raw = frame["quaternion"]
            if len(quaternion_raw) != 4:
                raise ValueError(f"Quaternion for frame {f_idx} in {pose_path.name} does not have 4 components")
            if args.quaternion_order == "wxyz":
                q_ordered = quaternion_raw
            else:
                q_ordered = [quaternion_raw[3], quaternion_raw[0], quaternion_raw[1], quaternion_raw[2]]
            q = normalize_quaternion(q_ordered)
            R_input = quaternion_to_matrix(q)
            if args.orientation == "camera_to_world":
                R_w2c = R_input.T
            else:
                R_w2c = R_input
            t = -R_w2c @ position.reshape(3, 1)
            q_w2c = matrix_to_quaternion(R_w2c)

            frame_idx_val = extract_frame_index(frame)
            if frame_idx_val is None:
                frame_idx_val = f_idx
            candidate_images = build_image_candidates(args, chosen_cam_token, frame_idx_val, frame.get("name"))
            image_path = resolve_image_path(images_root, candidate_images)
            if image_path is None:
                raise FileNotFoundError(
                    f"RGB image for frame {frame.get('name', f_idx)} (camera {chosen_cam_token}) not found. "
                    f"Looked for: {candidate_images}"
                )

            entries.append({
                "image_id": next_image_id,
                "camera_token": chosen_cam_token,
                "camera_colmap_id": camera_id_map[chosen_cam_token],
                "image_name": image_path.name,
                "image_source": image_path,
                "qvec": q_w2c,
                "tvec": t.reshape(-1),
                "frame_index": frame_idx_val,
            })
            next_image_id += 1

    entries.sort(key=lambda item: (item["frame_index"], item["camera_token"], item["image_id"]))

    if args.dry_run:
        print(f"Would create output at {output_root}")
        print("Cameras:")
        for cam_token, colmap_id in camera_id_map.items():
            intr = intrinsics_map.get(cam_token, default_intr)
            print(f"  COLMAP id {colmap_id}: source cam {cam_token} -> PINHOLE {intr}")
        print(f"Total images: {len(entries)}")
        for entry in entries[:5]:
            print(f"  Example image {entry['image_id']}: {entry['image_name']} -> q={entry['qvec']} t={entry['tvec']}")
        return

    sparse_dir = output_root / "sparse" / "0"
    images_dir = output_root / "images"
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    cameras_txt = sparse_dir / "cameras.txt"
    with cameras_txt.open("w", encoding="utf-8") as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam_token, colmap_id in sorted(camera_id_map.items(), key=lambda kv: kv[1]):
            intr = intrinsics_map.get(cam_token, default_intr)
            fh.write(
                f"{colmap_id} PINHOLE {int(intr['width'])} {int(intr['height'])} "
                f"{float(intr['fx']):.12f} {float(intr['fy']):.12f} {float(intr['cx']):.12f} {float(intr['cy']):.12f}\n"
            )

    images_txt = sparse_dir / "images.txt"
    with images_txt.open("w", encoding="utf-8") as fh:
        fh.write("# Image list with two lines of data per image:\n")
        fh.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for entry in entries:
            q = entry["qvec"]
            t = entry["tvec"]
            fh.write(
                f"{entry['image_id']} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} {entry['camera_colmap_id']} {entry['image_name']}\n"
            )
            fh.write("\n")

    points_txt = sparse_dir / "points3D.txt"
    with points_txt.open("w", encoding="utf-8") as fh:
        fh.write("# Empty point cloud placeholder.\n")

    points_bin = sparse_dir / "points3D.bin"
    with points_bin.open("wb") as fid:
        fid.write(struct.pack("<Q", 0))

    for entry in entries:
        dest = images_dir / entry["image_name"]
        if dest.exists():
            continue
        if args.copy_images:
            shutil.copy2(entry["image_source"], dest)
        else:
            rel_target = os.path.relpath(entry["image_source"], start=images_dir)
            if dest.exists():
                dest.unlink()
            os.symlink(rel_target, dest)

    print(f"Wrote COLMAP text model to {output_root}")
    print(f"  Cameras: {len(camera_id_map)}")
    print(f"  Images: {len(entries)}")


if __name__ == "__main__":
    main()
