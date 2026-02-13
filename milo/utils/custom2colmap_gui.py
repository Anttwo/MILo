import os
import struct
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2

# =========================================
# CONFIG 可根据需要修改
# =========================================
FX = 617.832096
FY = 621.743364
CX = 427.447679
CY = 233.692923
WIDTH, HEIGHT = 848, 480

END_ID = 1000

INTERVAL = 1                 # 稀疏化间隔
DT_THRESH = 0.05             # 平移阈值 (m)
DROT_THRESH = np.deg2rad(0)  # 旋转阈值(度→弧度)

MAX_DIST = 4.0               # 从 ply 中选择点的最大距离 (m)
VOXEL_SIZE = 0.01           # 最终全局点云体素下采样 (m)

OUT_DIR = "dataset"
SPARSE_DIR = "dataset/sparse/0"

def numeric_sort(files):
    return sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
# =========================================
# 读取PLY（ascii 或 binary）
# =========================================
def load_ply(filename):
    pcd = o3d.io.read_point_cloud(filename)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) * 255
    return pts, cols


# =========================================
# ORB-SLAM Tcw → Twc  Twc->Tcw
# =========================================
def Tcw_to_Twc(Rcw, tcw):
    Rwc = Rcw.T
    twc = -Rwc @ tcw
    return Rwc, twc

def Twc_to_Tcw(Rwc, twc):
    Rcw = Rwc.T
    tcw = -Rcw @ twc
    return Rcw, tcw



# pose difference
def pose_difference(T1, T2):
    R1, t1 = T1
    R2, t2 = T2
    dR = R2 @ R1.T
    drot = np.linalg.norm(R.from_matrix(dR).as_rotvec())
    dt = np.linalg.norm(t2 - t1)
    return dt, drot


# =========================================
# 写 cameras.bin / cameras.txt
# =========================================
def write_cameras(path_bin, path_txt):
    with open(path_bin, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<Q", WIDTH))
        f.write(struct.pack("<Q", HEIGHT))
        f.write(struct.pack("<dddd", FX, FY, CX, CY))

    with open(path_txt, "w") as f:
        f.write("# Camera list\n")
        f.write(f"1 PINHOLE {WIDTH} {HEIGHT} {FX} {FY} {CX} {CY}\n")


# =========================================
# 写 images.bin / images.txt
# =========================================
def write_images(path_bin, path_txt, rgb_paths, poses_cw):
    with open(path_bin, "wb") as f:
        f.write(struct.pack("<Q", len(rgb_paths)))

        for idx, rgb_path in enumerate(rgb_paths, 1):
            Rcw, tcw = poses_cw[idx - 1]
            q = R.from_matrix(Rcw).as_quat()
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]

            f.write(struct.pack("<i", idx))
            f.write(struct.pack("<dddd", qw, qx, qy, qz))
            f.write(struct.pack("<ddd", *tcw))
            f.write(struct.pack("<i", 1))
            f.write((os.path.basename(rgb_path) + "\0").encode())
            f.write(struct.pack("<Q", 0))

    with open(path_txt, "w") as f:
        for idx, rgb_path in enumerate(rgb_paths, 1):
            Rcw, tcw = poses_cw[idx - 1]
            q = R.from_matrix(Rcw).as_quat()
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]
            f.write(f"{idx} {qw} {qx} {qy} {qz} "
                    f"{tcw[0]} {tcw[1]} {tcw[2]} 1 "
                    f"{os.path.basename(rgb_path)}\n")


# =========================================
# 写 points3D.bin / points3D.txt / points3D.ply
# =========================================
def write_points(all_pts, all_cols, path_bin, path_txt, path_ply):

    # bin
    with open(path_bin, "wb") as f:
        f.write(struct.pack("<Q", len(all_pts)))
        for pid, (x, y, z), (r, g, b) in zip(range(1, len(all_pts)+1), all_pts, all_cols):
            f.write(struct.pack("<Q", pid))
            f.write(struct.pack("<ddd", x, y, z))
            f.write(struct.pack("<BBB", int(r), int(g), int(b)))
            f.write(struct.pack("<d", 1.0))
            f.write(struct.pack("<Q", 0))

    # txt
    with open(path_txt, "w") as f:
        for pid, (x, y, z), (r, g, b) in zip(range(1, len(all_pts)+1), all_pts, all_cols):
            f.write(f"{pid} {x} {y} {z} {int(r)} {int(g)} {int(b)} 1.0\n")

    # ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols / 255)
    o3d.io.write_point_cloud(path_ply, pcd)


# =========================================
# 可视化：轨迹 + 全局点云
# =========================================
def visualize(Twc_list, pts):
    cam_frames = []
    for Rwc, twc in Twc_list:
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        T = np.eye(4)
        T[:3, :3] = Rwc
        T[:3, 3] = twc
        cam_frame.transform(T)
        cam_frames.append(cam_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    o3d.visualization.draw_geometries([pcd] + cam_frames)


# =========================================
# MAIN PIPELINE
# =========================================
def main():

    # 读取输入
    rgb_files = numeric_sort(glob("../data/Inspection2Colmap/images/*.png"))
    ply_files = numeric_sort(glob("../data/Inspection2Colmap/ply/*.ply"))

    # 读取 ORB-SLAM Tcw
    Twc_list = []
    for line in open("../data/Inspection2Colmap/poses.txt"):
        nums = list(map(float, line.split()))
        T = np.array(nums).reshape(3, 4)
        Rwc, twc = T[:, :3], T[:, 3]
        Twc_list.append((Rwc, twc))
    # new branch

    # Tcw → Twc
    Tcw_list = [Twc_to_Tcw(Rwc, twc) for (Rwc, twc) in Twc_list]

    # =============================
    #  稀疏化（智能采样）
    # =============================
    selected = []
    last_pose = None

    for i in range(len(rgb_files)):
        if i > END_ID:
            continue
        if i % INTERVAL == 0:
            selected.append(i)
            last_pose = Twc_list[i]
            continue

        dt, drot = pose_difference(last_pose, Twc_list[i])
        if dt > DT_THRESH or drot > DROT_THRESH:
            selected.append(i)
            last_pose = Twc_list[i]

    selected = sorted(list(set(selected)))
    print("选中帧：", len(selected), "/", len(rgb_files))

    # 输出图像
    os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
    new_rgb_paths = []

    for new_id, old_id in enumerate(selected):
        img = cv2.imread(rgb_files[old_id])
        out = f"{OUT_DIR}/images/{new_id:06d}.png"
        cv2.imwrite(out, img)
        new_rgb_paths.append(out)

    selected_Tcw = [Tcw_list[i] for i in selected]
    selected_Twc = [Twc_list[i] for i in selected]

    # =============================
    #  构建 Points3D（含距离过滤）
    # =============================
    all_pts = []
    all_cols = []

    for i, old_id in enumerate(selected):
        Rcw, tcw = Tcw_list[old_id]
        Rwc, twc = Twc_list[old_id]

        pts, cols = load_ply(ply_files[old_id])

        # 距离过滤
        dist = np.linalg.norm(pts, axis=1)
        mask = dist < MAX_DIST
        pts = pts[mask]
        cols = cols[mask]

        # 相机 → 世界
        pts_w = (Rwc @ pts.T + twc.reshape(3, 1)).T

        all_pts.append(pts_w)
        all_cols.append(cols)

    all_pts = np.vstack(all_pts)
    all_cols = np.vstack(all_cols)

    print("点云合并后数量：", len(all_pts))

    # =============================
    #  体素化滤波 （减少点数量）
    # =============================
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols / 255)

    pcd = pcd.voxel_down_sample(VOXEL_SIZE)

    all_pts = np.asarray(pcd.points)
    all_cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    print("体素滤波后点数：", len(all_pts))

    # =============================
    #  输出 COLMAP 格式
    # =============================
    os.makedirs(SPARSE_DIR, exist_ok=True)

    write_cameras(f"{SPARSE_DIR}/cameras.bin", f"{SPARSE_DIR}/cameras.txt")
    write_images(f"{SPARSE_DIR}/images.bin", f"{SPARSE_DIR}/images.txt",
                 new_rgb_paths, selected_Tcw)

    write_points(all_pts, all_cols,
                 f"{SPARSE_DIR}/points3D.bin",
                 f"{SPARSE_DIR}/points3D.txt",
                 f"{SPARSE_DIR}/points3D.ply")

    # =============================
    #  可视化（相机轨迹 + 全局点云）
    # =============================
    visualize(selected_Twc, all_pts)

    print("全部完成！")


if __name__ == "__main__":
    main()
