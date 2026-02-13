import os
import struct
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
import cv2

###############################################################
# 配置区（按需修改）
###############################################################

# 相机内参
fx = 617.832096
fy = 621.743364
cx = 427.447679
cy = 233.692923
WIDTH, HEIGHT = 848, 480

# 点云采样参数（你需要的“按距离阈值选点”）
DIST_MIN = 0.2             # 最小深度
DIST_MAX = 4.5             # 最大深度
POINTS_PER_FRAME = 280000   # 每帧最多采多少点（最近排序后取）

# 输入目录
RGB_DIR = "../data/Inspection2Colmap/images"  # 输入 RGB 图像
PLY_DIR = "../data/Inspection2Colmap/ply"      # 输入每帧点云
POSE_FILE = "../data/Inspection2Colmap/poses.txt"   # ORB-SLAM 输出的 T_cw（world→camera）

# 输出目录
OUT_DIR = "dataset"
OUT_IMG_DIR = f"{OUT_DIR}/images"
OUT_SPARSE = f"{OUT_DIR}/sparse/0"

def numeric_sort(files):
    return sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
###############################################################
# 工具函数：读取 PLY（xyzrgb）
###############################################################
def load_ply(filename):
    with open(filename, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == "end_header":
                break

        is_binary = any("binary" in h for h in header)
        props = [h for h in header if h.startswith("property")]
        prop_names = [p.split()[-1] for p in props]
        has_color = all(c in prop_names for c in ["red", "green", "blue"])

        for h in header:
            if h.startswith("element vertex"):
                n_pts = int(h.split()[-1])

        data = []

        if is_binary:
            for _ in range(n_pts):
                x, y, z = struct.unpack("fff", f.read(12))
                if has_color:
                    r, g, b = struct.unpack("BBB", f.read(3))
                else:
                    r = g = b = 255
                data.append([x, y, z, r, g, b])
        else:
            for _ in range(n_pts):
                tokens = f.readline().decode('utf-8').split()
                x, y, z = map(float, tokens[:3])
                if has_color:
                    r, g, b = map(int, tokens[3:6])
                else:
                    r = g = b = 255
                data.append([x, y, z, r, g, b])

    return np.array(data)


###############################################################
# 写 COLMAP bin/txt
###############################################################
def write_cameras_bin_txt(path_bin, path_txt):
    os.makedirs(os.path.dirname(path_bin), exist_ok=True)

    # bin
    with open(path_bin, "wb") as f:
        f.write(struct.pack("<Q", 1))  # number of cameras
        f.write(struct.pack("<i", 1))  # camera id
        f.write(struct.pack("<i", 1))  # model = PINHOLE
        f.write(struct.pack("<Q", WIDTH))
        f.write(struct.pack("<Q", HEIGHT))
        f.write(struct.pack("<dddd", fx, fy, cx, cy))

    # txt
    with open(path_txt, "w") as f:
        f.write("# Camera list\n")
        f.write(f"1 PINHOLE {WIDTH} {HEIGHT} {fx} {fy} {cx} {cy}\n")


def write_images_bin_txt(path_bin, path_txt, rgb_paths, poses_cw):
    # bin
    with open(path_bin, "wb") as f:
        f.write(struct.pack("<Q", len(rgb_paths)))

        for img_id, img_path in enumerate(rgb_paths, 1):
            Rcw, tcw = poses_cw[img_id - 1]

            q = R.from_matrix(Rcw).as_quat()  # x,y,z,w
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]

            f.write(struct.pack("<i", img_id))
            f.write(struct.pack("<dddd", qw, qx, qy, qz))
            f.write(struct.pack("<ddd", *tcw))
            f.write(struct.pack("<i", 1))  # camera ID
            f.write((os.path.basename(img_path) + "\0").encode())
            f.write(struct.pack("<Q", 0))  # no keypoints

    # txt
    with open(path_txt, "w") as f:
        for img_id, img_path in enumerate(rgb_paths, 1):
            Rcw, tcw = poses_cw[img_id - 1]
            q = R.from_matrix(Rcw).as_quat()
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]

            f.write(f"{img_id} {qw} {qx} {qy} {qz} "
                    f"{tcw[0]} {tcw[1]} {tcw[2]} 1 "
                    f"{os.path.basename(img_path)}\n")


def write_points3D_bin_txt(path_bin, path_txt, pts, cols):
    # bin
    with open(path_bin, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for pid, (x, y, z), (r, g, b) in zip(range(1, len(pts) + 1), pts, cols):
            f.write(struct.pack("<Q", pid))
            f.write(struct.pack("<ddd", x, y, z))
            f.write(struct.pack("<BBB", int(r), int(g), int(b)))
            f.write(struct.pack("<d", 1.0))
            f.write(struct.pack("<Q", 0))

    # txt
    with open(path_txt, "w") as f:
        for pid, (x, y, z), (r, g, b) in zip(range(1, len(pts) + 1), pts, cols):
            f.write(f"{pid} {x} {y} {z} {int(r)} {int(g)} {int(b)} 1.0\n")


###############################################################
# 主流程
###############################################################
def main():
    rgb_files = numeric_sort(glob(f"{RGB_DIR}/*.png"))
    ply_files = numeric_sort(glob(f"{PLY_DIR}/*.ply"))

    print(f"加载 {len(rgb_files)} 张图像，{len(ply_files)} 个点云...")

    # 读取 ORB-SLAM 的 T_cw (world→camera)
    poses_cw = []
    for line in open(POSE_FILE):
        nums = list(map(float, line.split()))
        T = np.array(nums).reshape(3, 4)
        Rcw = T[:, :3]
        tcw = T[:, 3]
        poses_cw.append((Rcw, tcw))

    # 输出目录
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_SPARSE, exist_ok=True)

    # 拷贝图像到 dataset/images，并重新编号
    new_rgb_paths = []
    for i, src in enumerate(rgb_files):
        dst = f"{OUT_IMG_DIR}/{i:06d}.png"
        cv2.imwrite(dst, cv2.imread(src))
        new_rgb_paths.append(dst)

    ###############################################################
    # 构建 points3D：按距离从每帧选取点（你要求的部分）
    ###############################################################
    all_pts_world = []
    all_cols_world = []

    for i, ply in enumerate(ply_files):
        print(f"[{i+1}/{len(ply_files)}] 处理点云 {ply}")

        Rcw, tcw = poses_cw[i]
        Rwc = Rcw.T
        twc = -Rwc @ tcw

        data = load_ply(ply)

        Xc = data[:, :3]
        C = data[:, 3:]

        # 按距离过滤点
        dist = np.linalg.norm(Xc, axis=1)
        mask = (dist > DIST_MIN) & (dist < DIST_MAX)
        Xc = Xc[mask]
        C = C[mask]
        dist = dist[mask]

        # 最多取 POINTS_PER_FRAME 个最近点
        if len(Xc) > POINTS_PER_FRAME:
            idx = np.argsort(dist)[:POINTS_PER_FRAME]
            Xc = Xc[idx]
            C = C[idx]

        # 转世界坐标
        Xc_t = Xc.T
        Xw = (Rwc @ Xc_t + twc.reshape(3, 1)).T

        all_pts_world.append(Xw)
        all_cols_world.append(C)

    # 合并所有帧点
    all_pts_world = np.vstack(all_pts_world)
    all_cols_world = np.vstack(all_cols_world)

    print(f"最终点云数量：{len(all_pts_world)}")

    ###############################################################
    # 写入 COLMAP 格式
    ###############################################################
    write_cameras_bin_txt(
        f"{OUT_SPARSE}/cameras.bin",
        f"{OUT_SPARSE}/cameras.txt"
    )

    write_images_bin_txt(
        f"{OUT_SPARSE}/images.bin",
        f"{OUT_SPARSE}/images.txt",
        new_rgb_paths,
        poses_cw
    )

    write_points3D_bin_txt(
        f"{OUT_SPARSE}/points3D.bin",
        f"{OUT_SPARSE}/points3D.txt",
        all_pts_world,
        all_cols_world
    )

    print("\n输出完成！COLMAP + Gaussian Splatting 数据集已生成到 dataset/")


if __name__ == "__main__":
    main()
