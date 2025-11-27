#!/usr/bin/env python3
"""
Verification script to check camera pose interpretation.
This script loads the camera poses and PLY file to verify if Gaussians
are in front of or behind the cameras.
"""

import json
import numpy as np
from pathlib import Path


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    rotation = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return rotation


def load_ply_points(ply_path):
    """Load point positions from PLY file."""
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return positions


def check_camera_pose_current(camera_entry, points):
    """Check with CURRENT (incorrect) pose interpretation."""
    quaternion = camera_entry["quaternion"]
    camera_center = np.array(camera_entry["position"], dtype=np.float32)

    # Current code (WRONG according to codex):
    rotation = quaternion_to_rotation_matrix(quaternion)
    translation = -rotation.T @ camera_center

    # Transform points to camera space
    # With current interpretation: R is passed as-is, T is translation
    # getWorld2View2 expects R to be C2W and builds W2C as [R.T | T]
    # So W2C rotation is rotation.T, W2C translation is translation
    R_w2c = rotation.T
    t_w2c = translation

    # Transform points: p_cam = R_w2c @ p_world + t_w2c
    points_cam = (R_w2c @ points.T).T + t_w2c

    # Check z coordinate (positive = in front of camera)
    in_front = np.sum(points_cam[:, 2] > 0)
    total = len(points)
    fraction = in_front / total

    return fraction, in_front, total


def check_camera_pose_corrected(camera_entry, points):
    """Check with CORRECTED pose interpretation."""
    quaternion = camera_entry["quaternion"]
    camera_center = np.array(camera_entry["position"], dtype=np.float32)

    # Corrected code (as suggested by codex):
    rotation_w2c = quaternion_to_rotation_matrix(quaternion)
    rotation_c2w = rotation_w2c.T
    translation_w2c = -rotation_w2c @ camera_center

    # With corrected interpretation: R_c2w is passed, T is translation_w2c
    # getWorld2View2 builds W2C as [R_c2w.T | T] = [R_w2c | translation_w2c]
    R_w2c = rotation_c2w.T  # = rotation_w2c
    t_w2c = translation_w2c

    # Transform points: p_cam = R_w2c @ p_world + t_w2c
    points_cam = (R_w2c @ points.T).T + t_w2c

    # Check z coordinate (positive = in front of camera)
    in_front = np.sum(points_cam[:, 2] > 0)
    total = len(points)
    fraction = in_front / total

    return fraction, in_front, total


def main():
    # Paths
    camera_poses_path = Path("/milo/data/bridge_small/camera_poses_cam1.json")
    ply_path = Path("/milo/data/bridge_small/yufu_bridge_small.ply")

    # Load data
    print(f"Loading camera poses from {camera_poses_path}")
    with open(camera_poses_path, 'r') as f:
        camera_entries = json.load(f)
    print(f"Loaded {len(camera_entries)} cameras")

    print(f"\nLoading point cloud from {ply_path}")
    points = load_ply_points(ply_path)
    print(f"Loaded {len(points)} points")

    # Check first camera with both interpretations
    print("\n" + "="*80)
    print("CHECKING CAMERA 0 (traj_0_cam0)")
    print("="*80)

    camera_0 = camera_entries[0]
    print(f"Camera position: {camera_0['position']}")
    print(f"Camera quaternion: {camera_0['quaternion']}")

    print("\n--- Current (WRONG) interpretation ---")
    frac_current, in_front_current, total = check_camera_pose_current(camera_0, points)
    print(f"Points in front of camera: {in_front_current}/{total} ({frac_current*100:.2f}%)")

    print("\n--- Corrected interpretation ---")
    frac_corrected, in_front_corrected, total = check_camera_pose_corrected(camera_0, points)
    print(f"Points in front of camera: {in_front_corrected}/{total} ({frac_corrected*100:.2f}%)")

    # Check a few more cameras
    print("\n" + "="*80)
    print("CHECKING FIRST 5 CAMERAS")
    print("="*80)
    print(f"{'Camera':<15} {'Current (wrong)':<20} {'Corrected':<20}")
    print("-" * 80)

    for i in range(min(5, len(camera_entries))):
        camera = camera_entries[i]
        frac_current, _, _ = check_camera_pose_current(camera, points)
        frac_corrected, _, _ = check_camera_pose_corrected(camera, points)
        print(f"{camera['name']:<15} {frac_current*100:>6.2f}%            {frac_corrected*100:>6.2f}%")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    if frac_current < 0.5 and frac_corrected > 0.5:
        print("✓ Codex's analysis is CORRECT!")
        print("  - Current code: Most points are BEHIND cameras (wrong)")
        print("  - Corrected code: Most points are IN FRONT of cameras (correct)")
        print("\nThe quaternions should be interpreted as world→camera rotations,")
        print("and the fix suggested by codex is needed.")
    else:
        print("✗ Results don't match codex's analysis.")
        print("  Further investigation needed.")


if __name__ == "__main__":
    main()