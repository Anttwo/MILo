#!/usr/bin/env python3
# clean_convert_mesh.py
# Usage:
#   python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply
#   python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --simplify 300000
#   python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --keep-components 0.02 --no-glb

import argparse
import os
import sys
from pathlib import Path

# deps: pip install pymeshlab trimesh plyfile
try:
    import pymeshlab as ml
except Exception as e:
    print("[ERR] pymeshlab not available. `pip install pymeshlab` first.\n", e)
    sys.exit(1)

try:
    import trimesh
except Exception as e:
    print("[WARN] trimesh not available. GLB export will be disabled.\n", e)
    trimesh = None


def human(n: int) -> str:
    return f"{n:,}"


def bbox_diag_from_ms(ms: ml.MeshSet) -> float:
    """Compute bounding-box diagonal length from current mesh (in MeshSet)."""
    vm = ms.current_mesh().vertex_matrix()
    if vm.size == 0:
        return 0.0
    vmin = vm.min(axis=0)
    vmax = vm.max(axis=0)
    return float(((vmax - vmin) ** 2).sum() ** 0.5)


def print_stats(tag: str, ms: ml.MeshSet):
    m = ms.current_mesh()
    print(f"{tag:>10} | V={human(m.vertex_number())}  F={human(m.face_number())}")


def clean_mesh(ms: ml.MeshSet, keep_components_frac: float):
    """
    温和清理，不破坏整体拓扑：
      - 去重顶点/面、移除未引用顶点
      - 修复/去除非流形边（依版本自动选择）
      - 依据直径阈值移除小浮块
      - 重新计算法线
    """
    # 基础去重/引用修复
    if hasattr(ms, "meshing_remove_duplicate_faces"):
        ms.meshing_remove_duplicate_faces()
    if hasattr(ms, "meshing_remove_duplicate_vertices"):
        ms.meshing_remove_duplicate_vertices()
    if hasattr(ms, "meshing_remove_unreferenced_vertices"):
        ms.meshing_remove_unreferenced_vertices()

    # 非流形边：不同版本函数名不一致，这里做兼容
    if hasattr(ms, "meshing_repair_non_manifold_edges"):
        ms.meshing_repair_non_manifold_edges()
    elif hasattr(ms, "meshing_remove_non_manifold_edges"):
        ms.meshing_remove_non_manifold_edges()

    # 小组件/浮块移除（以整体 bbox 对角线为参照）
    diag = bbox_diag_from_ms(ms)
    if keep_components_frac > 0 and diag > 0 and hasattr(ms, "meshing_remove_isolated_pieces_wrt_diameter"):
        thr = float(diag * keep_components_frac)  # 绝对长度阈值
        ms.meshing_remove_isolated_pieces_wrt_diameter(mincomponentdiag=thr)

    # 法线
    if hasattr(ms, "compute_normals_for_point_sets"):
        ms.compute_normals_for_point_sets()
    if hasattr(ms, "compute_normals_for_meshes"):
        ms.compute_normals_for_meshes()



def simplify_mesh(ms: ml.MeshSet, target_tris: int):
    """
    Quadric decimation（网格简化），保守设置；如果当前面数已小于目标则跳过。
    """
    current_f = ms.current_mesh().face_number()
    if target_tris <= 0 or current_f <= target_tris:
        print(f"[INFO] Skip simplify: current F={human(current_f)} <= target {human(max(target_tris,0))}")
        return

    print(f"[INFO] Simplifying: {human(current_f)} → {human(target_tris)} (quadric)")
    # 尽量保持边界与法线
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_tris,
        preservenormal=True,
        preservetopology=True,
        qualitythr=0.3,            # 较保守
        optimalplacement=True,
        planarquadric=True,
        autoclean=True
    )
    ms.compute_normals_for_meshes()


def export_all(ms: ml.MeshSet, out_dir: Path, stem: str, do_ply: bool, do_obj: bool, do_glb: bool):
    """
    为了兼容不同版本的 pymeshlab，这里保存时不传任何可选参数。
    先保存 PLY/OBJ（如果启用），然后用 trimesh 从 PLY 导出 GLB。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"{stem}.ply"
    obj_path = out_dir / f"{stem}.obj"
    glb_path = out_dir / f"{stem}.glb"

    # 1) 保存 PLY（建议启用，后面 GLB 也依赖它）
    if do_ply:
        ms.save_current_mesh(str(ply_path))  # 不传 save_* 参数以避免版本差异
        print(f"[OK] Saved: {ply_path}")

    # 2) 保存 OBJ（同样不带可选参数；注意 OBJ 不支持存储逐顶点颜色）
    if do_obj:
        ms.save_current_mesh(str(obj_path))
        print(f"[OK] Saved: {obj_path}")

    # 3) 导出 GLB（走 trimesh）
    if do_glb:
        if trimesh is None:
            print("[WARN] trimesh not installed; skip GLB.")
            return

        # 若用户禁用了 PLY 导出，则先写一个中间 PLY 供 trimesh 读取
        tmp_ply = ply_path
        need_tmp = False
        if not do_ply or not tmp_ply.exists():
            tmp_ply = out_dir / f"{stem}__tmp_for_glb.ply"
            ms.save_current_mesh(str(tmp_ply))
            need_tmp = True

        tm = trimesh.load(str(tmp_ply), process=False)
        tm.export(str(glb_path))
        print(f"[OK] Saved: {glb_path}")

        # 可选：清理临时文件
        if need_tmp and tmp_ply.exists():
            try:
                tmp_ply.unlink()
            except Exception:
                pass



def main():
    ap = argparse.ArgumentParser(description="Clean and convert MILo mesh_learnable_sdf.ply to PLY/OBJ/GLB.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input PLY path (e.g., ./output/Ignatius/mesh_learnable_sdf.ply)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: same as input)")
    ap.add_argument("--stem", default="mesh_clean", help="Output filename stem (default: mesh_clean)")
    ap.add_argument("--keep-components", type=float, default=0.02,
                    help="Remove small isolated components with diameter < frac * bbox_diag (default: 0.02)")
    ap.add_argument("--simplify", type=int, default=0,
                    help="Target triangle count for decimation (0=disable). Example: 300000")
    ap.add_argument("--no-ply", action="store_true", help="Do not export PLY")
    ap.add_argument("--no-obj", action="store_true", help="Do not export OBJ")
    ap.add_argument("--no-glb", action="store_true", help="Do not export GLB")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"[ERR] Input not found: {in_path}")
        sys.exit(2)

    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    do_ply = not args.no_ply
    do_obj = not args.no_obj
    do_glb = not args.no_glb

    ms = ml.MeshSet()
    try:
        ms.load_new_mesh(str(in_path))
    except Exception as e:
        print(f"[ERR] Failed to load mesh: {in_path}\n{e}")
        sys.exit(3)

    # 基本类型检查
    m = ms.current_mesh()
    if m.face_number() == 0:
        print("[ERR] The input PLY has 0 faces (looks like a point cloud). Aborting.")
        sys.exit(4)

    print_stats("Loaded", ms)

    # 清理
    clean_mesh(ms, keep_components_frac=args.keep_components)
    print_stats("Cleaned", ms)

    # 可选简化
    if args.simplify > 0:
        simplify_mesh(ms, target_tris=args.simplify)
        print_stats("Simplify", ms)

    # 导出
    export_all(ms, out_dir=out_dir, stem=args.stem, do_ply=do_ply, do_obj=do_obj, do_glb=do_glb)

    print("\n[DONE] All good.")


if __name__ == "__main__":
    main()
