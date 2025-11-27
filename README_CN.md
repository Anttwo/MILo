# MILo_rtx50 — CUDA 12.8 / RTX 50 系列本地编译与运行记录（Ubuntu 24.04 + uv + PyTorch 2.7.1+cu128）

> 本 README 记录我们在 **RTX 50 系列 + CUDA 12.8** 环境下，fork 项目 [Anttwo/MILo](https://github.com/Anttwo/MILo) 的**本地编译适配、关键修改与可复现实验步骤**。
> 目标：无需 Conda，使用 **uv + venv** 完成子模块编译与训练、网格提取、渲染和评测。

---

## 环境

- OS：Ubuntu 24.04
- GPU：RTX 50 系列（Blackwell）
- CUDA Toolkit：12.8（NVCC `/usr/local/cuda-12.8/bin/nvcc`）
- Python：3.12.3（venv 管理，包管理用 **uv**）
- PyTorch：**2.7.1+cu128**（官方二进制，**C++11 ABI=1**）
- C/C++：GCC 13.3
- CMake：系统版本（apt）
- 重要环境变量（训练/提取/渲染时常用）：
  ```bash
  export NVDIFRAST_BACKEND=cuda
  export TORCH_CUDA_ARCH_LIST="12.0+PTX"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32,garbage_collection_threshold:0.6"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  # Mesh 正则化网格分辨率缩放，越小越省显存
  export MILO_MESH_RES_SCALE=0.3
  # （可选）按三角形分块的大小，缓解 nvdiffrast CUDA 后端显存峰值
  export MILO_RAST_TRI_CHUNK=150000
  ```

---

## Submodules 安装

> 这些我们已验证可在 CUDA 12.8 + PyTorch 2.7.1 下成功编译/安装。

### 1) 安装 Gaussian Splatting 子模块（通过 **pip**）

```bash
pip install submodules/diff-gaussian-rasterization_ms
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization_gof
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

> 备注：`nvdiffrast` 走 **JIT 编译**（运行时由 PyTorch cpp_extension 触发）。
> 若选择 **OpenGL(GL) 后端**，需要系统头：`sudo apt install -y libegl-dev libopengl-dev libgles2-mesa-dev ninja-build`。
> 我们为省事改用 **CUDA 后端**：`export NVDIFRAST_BACKEND=cuda`（无需 EGL 头）。

### 2) 安装 `tetra_triangulation` 的系统依赖（Delaunay 三角剖分）

原项目用 **conda** 安装系统级 C/C++ 依赖（cmake/gmp/cgal）。由于我们用 **uv** 只管 Python 包，需要通过 **apt**（系统包管理器）安装这些 C/C++ 库：

```bash
# 用 apt 安装 C/C++ 依赖（Ubuntu 24.04）
sudo apt update
sudo apt install -y \
  build-essential \
  cmake ninja-build \
  libgmp-dev libmpfr-dev libcgal-dev \
  libboost-all-dev

# （可选）可能用到：
# sudo apt install -y libeigen3-dev
```

**说明：**
- `libcgal-dev` 提供 CGAL 头文件（Ubuntu 24.04 上主要是 header-only）
- `libgmp-dev` 和 `libmpfr-dev` 是 CGAL 的数值后端
- **uv 仅负责 Python 侧依赖**；像 CGAL/GMP/MPFR 这种 C/C++ 依赖必须走系统包管理器（apt、brew、pacman）
- **macOS**：`brew install cmake cgal gmp mpfr boost`
- **Arch Linux**：`sudo pacman -S cgal gmp mpfr boost cmake base-devel`

### 3) 编译 `tetra_triangulation` 并对齐 ABI

> **重要：** 该模块需要与 PyTorch 2.7.1（C++11 ABI=1）对齐 ABI。我们使用头文件方式来强制这一点。

**a) 创建 ABI 强制头文件：**

创建 `submodules/tetra_triangulation/src/force_abi.h`：
```cpp
#pragma once
// 在任何 STL 头之前强制新 ABI
#if defined(_GLIBCXX_USE_CXX11_ABI)
#  undef _GLIBCXX_USE_CXX11_ABI
#endif
#define _GLIBCXX_USE_CXX11_ABI 1
```

**b) 修改源文件：**

在以下文件的**第一行**加入 `#include "force_abi.h"`：
- `submodules/tetra_triangulation/src/py_binding.cpp`
- `submodules/tetra_triangulation/src/triangulation.cpp`

**c) 构建与安装：**

```bash
cd submodules/tetra_triangulation
rm -rf build CMakeCache.txt CMakeFiles tetranerf/utils/extension/tetranerf_cpp_extension*.so

# 指向当前 PyTorch 的 CMake 前缀/动态库路径
export CMAKE_PREFIX_PATH="$(python - <<'PY'
import torch; print(torch.utils.cmake_prefix_path)
PY
)"
export TORCH_LIB_DIR="$(python - <<'PY'
import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" .
cmake --build . -j"$(nproc)"

# 安装（可选，便于可编辑引用）
uv pip install -e .
cd ../../
```

> **说明：** 如需排查 ABI 问题，请参阅下方**关键问题 1**部分。

---

## 关键问题 1：`tetra_triangulation` ABI 不匹配（已解决）

**现象**
运行 `from tetranerf.utils import extension as ext` 报错：

```
undefined symbol: _ZN3c106detail14torchCheckFailEPKcS2_jRKSs
```

尾部 `RKSs` 表示 **老 ABI（_GLIBCXX_USE_CXX11_ABI=0）**，而我们的 PyTorch 2.7.1 使用 **新 ABI（=1）**。

**修复**
在 `submodules/tetra_triangulation` 中加入强制 ABI 的头文件，**稳定锁定 ABI=1**：

* 创建文件：`src/force_abi.h`

  ```cpp
  #pragma once
  // 在任何 STL 头之前强制新 ABI
  #if defined(_GLIBCXX_USE_CXX11_ABI)
  #  undef _GLIBCXX_USE_CXX11_ABI
  #endif
  #define _GLIBCXX_USE_CXX11_ABI 1
  ```

* 修改：在 `src/py_binding.cpp` 与 `src/triangulation.cpp` 的**第一行**加入

  ```cpp
  #include "force_abi.h"
  ```

> **说明：** 使用这种头文件方式即可强制 ABI=1，无需额外修改 CMakeLists.txt。

**构建命令（就地 in-source，产物落到包路径）**

```bash
cd submodules/tetra_triangulation
rm -rf build CMakeCache.txt CMakeFiles tetranerf/utils/extension/tetranerf_cpp_extension*.so

# 指向当前 PyTorch 的 CMake 前缀/动态库路径
export CMAKE_PREFIX_PATH="$(python - <<'PY'
import torch; print(torch.utils.cmake_prefix_path)
PY
)"
export TORCH_LIB_DIR="$(python - <<'PY'
import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" .
cmake --build . -j"$(nproc)"

# 安装（可选，便于可编辑引用）
uv pip install -e .
```

---

## 关键问题 2：nvdiffrast GL 后端缺少 `EGL/egl.h`（已绕过）

* 方案 A：`sudo apt install -y libegl-dev libopengl-dev libgles2-mesa-dev` 后继续用 GL。
* **方案 B（我们采用）**：切到 **CUDA 后端**：`export NVDIFRAST_BACKEND=cuda`，不依赖 EGL 头。

---

## 关键问题 3：nvdiffrast CUDA 后端显存 OOM（已解决）

**现象**
`cudaMalloc(&m_gpuPtr, bytes)` OOM（error: 2），尤其在 Mesh 正则化阶段。

**修复（两点）**

1. 在 `milo/scene/mesh.py` 中替换 `nvdiff_rasterization` 实现：

   * 支持**按三角形分块**（环境变量 `MILO_RAST_TRI_CHUNK` 指定块大小）
   * **修正 CUDA 后端的 `ranges` 必须在 CPU**（`dr.rasterize(..., ranges=<CPU tensor>)`）

   <details>
   <summary>修改后的函数（点击展开）</summary>

   ```python
   def nvdiff_rasterization(
       camera,
       image_height: int,
       image_width: int,
       verts: torch.Tensor,
       faces: torch.Tensor,
       return_indices_only: bool = False,
       glctx=None,
       return_rast_out: bool = False,
       return_positions: bool = False,
   ):
       """
       与原函数等价的替换版，支持按三角形分块（env: MILO_RAST_TRI_CHUNK），
       并修正：nvdiffrast CUDA 后端的 `ranges` 必须在 CPU。
       """
       import os
       import torch
       import nvdiffrast.torch as dr

       device = verts.device
       dtype = verts.dtype

       cam_mtx = camera.full_proj_transform
       pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device, dtype=dtype)], dim=1)
       pos = torch.matmul(pos, cam_mtx)[None]  # [1,V,4]

       faces = faces.to(torch.int32).contiguous()
       faces_dev = faces.to(pos.device)

       H, W = int(image_height), int(image_width)
       chunk = int(os.getenv("MILO_RAST_TRI_CHUNK", "0") or "0")
       use_chunking = chunk > 0 and faces.shape[0] > chunk

       if not use_chunking:
           rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces_dev, resolution=[H, W])
           bary_coords = rast_out[..., :2]
           zbuf = rast_out[..., 2]
           pix_to_face = rast_out[..., 3].to(torch.int32) - 1
           if return_indices_only:
               return pix_to_face
           _out = (bary_coords, zbuf, pix_to_face)
           if return_rast_out:
               _out += (rast_out,)
           if return_positions:
               _out += (pos,)
           return _out

       z_ndc = (pos[..., 2:3] / (pos[..., 3:4] + 1e-20)).contiguous()

       best_rast, best_depth = None, None
       n_faces, start = int(faces.shape[0]), 0

       def _normalize_tri_id(rast_chunk, start_idx, count_idx):
           tri_raw = rast_chunk[..., 3:4].to(torch.int64)
           if tri_raw.numel() == 0:
               return rast_chunk[..., 3:4]
           maxid = int(tri_raw.max().item())
           if maxid == 0:
               return rast_chunk[..., 3:4]
           if maxid <= count_idx:
               tri_adj = torch.where(tri_raw > 0, tri_raw + start_idx, tri_raw)
           else:
               tri_adj = tri_raw
           return tri_adj.to(rast_chunk.dtype)

       while start < n_faces:
           count = min(chunk, n_faces - start)
           # ranges 必须在 CPU
           ranges_cpu = torch.tensor([[start, count]], device="cpu", dtype=torch.int32)

           rast_chunk, _ = dr.rasterize(glctx, pos=pos, tri=faces_dev, resolution=[H, W], ranges=ranges_cpu)
           depth_chunk, _ = dr.interpolate(z_ndc, rast_chunk, faces_dev)
           tri_id_adj = _normalize_tri_id(rast_chunk, start, count)

           if best_rast is None:
               best_rast = torch.zeros_like(rast_chunk)
               best_depth = torch.full_like(depth_chunk, float("inf"))

           hit = (tri_id_adj > 0)
           prev_hit = (best_rast[..., 3:4] > 0)
           closer = hit & (~prev_hit | (depth_chunk < best_depth))

           rast_chunk = torch.cat([rast_chunk[..., :3], tri_id_adj], dim=-1)

           best_depth = torch.where(closer, depth_chunk, best_depth)
           best_rast = torch.where(closer.expand_as(best_rast), rast_chunk, best_rast)

           start += count

       rast_out = best_rast
       bary_coords = rast_out[..., :2]
       zbuf = rast_out[..., 2]
       pix_to_face = rast_out[..., 3].to(torch.int32) - 1

       if return_indices_only:
           return pix_to_face

       _output = (bary_coords, zbuf, pix_to_face)
       if return_rast_out:
           _output += (rast_out,)
       if return_positions:
           _output += (pos,)
       return _output
   ```

   </details>

2. 运行时降低内存峰值：

   * `MILO_MESH_RES_SCALE=0.3`（mesh 正则化分辨率缩放）
   * `MILO_RAST_TRI_CHUNK=150000`（三角形分块大小）
   * `--data_device cpu`（相机/数据在 CPU）

---

## 复现实验步骤（Ignatius）

> 数据路径：`./data/Ignatius`
> 输出目录：`./output/Ignatius`

### 1) 训练

```bash
cd milo
export NVDIFRAST_BACKEND=cuda
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32,garbage_collection_threshold:0.6"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MILO_MESH_RES_SCALE=0.3
export MILO_RAST_TRI_CHUNK=150000

python train.py -s ./data/Ignatius -m ./output/Ignatius \
  --imp_metric outdoor \
  --rasterizer gof \
  --mesh_config verylowres \
  --sampling_factor 0.2 \
  --data_device cpu \
  --log_interval 200
```

**产出**（位于 `./output/Ignatius`）

* 训练好的场景（Gaussians + learnable SDF 与 mesh 正则化状态等）
* 日志与中间文件（按脚本配置，控制台打印训练进度）

### 2) 网格提取（SDF）

```bash
python mesh_extract_sdf.py \
  -s ./data/Ignatius \
  -m ./output/Ignatius \
  --rasterizer gof \
  --config verylowres \
  --data_device cpu
```

**产出**

* **`./output/Ignatius/mesh_learnable_sdf.ply`**（已确认可在 MeshLab 正常打开）

### 3) 渲染

```bash
python render.py \
  -m ./output/Ignatius \
  -s ./data/Ignatius \
  --rasterizer gof \
  --eval
```

**产出**

* 渲染图像（训练/测试视角），保存到模型输出目录中的渲染子目录（以脚本实际打印为准）

### 4) 图像指标

```bash
python metrics.py -m ./output/Ignatius
```

**产出**

* 控制台输出 PSNR/SSIM（以及仓库脚本保存的对应文件，位于模型目录下；以实际实现为准）

### 5) PLY 格式转换（可选）

提取的 PLY 网格可以使用 `clean_convert_mesh.py` 脚本转换为其他常用的 3D 格式（OBJ/GLB）以便在各种 3D 软件中使用。脚本同时提供网格清理功能作为可选项。

**安装额外依赖**
```bash
pip install pymeshlab trimesh plyfile
```

**基本用法**
```bash
# 基本转换（输出 PLY、OBJ、GLB）
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply

# 转换并简化到 30 万三角形
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --simplify 300000

# 转换时清理小组件（默认 0.02 = 移除直径小于 2% bbox 对角线的组件）
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --keep-components 0.02

# 只输出特定格式
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --no-glb  # 不输出 GLB
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --no-obj  # 不输出 OBJ

# 指定输出目录和文件名
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply \
  --out-dir ./output/Ignatius/converted \
  --stem mesh_final
```

**主要功能**
- **格式转换**：将 PLY 转换为 OBJ、GLB 格式（适合不同 3D 软件和 Web 展示）
- **可选清理**：去除重复顶点/面、修复非流形边、移除小浮块
- **可选简化**：基于 Quadric decimation 的保形简化

**产出**（默认保存在输入文件同目录）
* `mesh_clean.ply` - 转换后的 PLY 网格（带顶点颜色）
* `mesh_clean.obj` - OBJ 格式（注意：OBJ 不支持顶点颜色）
* `mesh_clean.glb` - GLB 格式（适合 Web 展示和 Blender/Unity 等软件导入）

---

## 我们做了什么修改（相对上游）

1. **`submodules/tetra_triangulation`**

   * 新增 `src/force_abi.h`，并在 `src/py_binding.cpp` 和 `src/triangulation.cpp` 首行 `#include "force_abi.h"`：**强制使用 C++11 新 ABI (=1)**

2. **`milo/scene/mesh.py`**

   * 替换 `nvdiff_rasterization`：

     * 支持 **MILO_RAST_TRI_CHUNK** 三角形分块
     * **修正 CUDA 后端 `ranges` 必须是 CPU Tensor**
     * 其余行为与原函数保持一致

3. **运行配置**

   * 默认使用 nvdiffrast **CUDA** 后端（`NVDIFRAST_BACKEND=cuda`），规避 EGL 依赖
   * 为 Blackwell 指定 `TORCH_CUDA_ARCH_LIST="12.0+PTX"`
   * 降低峰值显存：`MILO_MESH_RES_SCALE=0.3`、`MILO_RAST_TRI_CHUNK=150000`、`--data_device cpu`

---

## 常见问题与排障

* **`undefined symbol: ... torchCheckFail ... RKSs`**
  这是 ABI=0 的符号；请按上面的补丁重编 `tetra_triangulation`。

* **`fatal error: EGL/egl.h: No such file or directory`**
  若坚持 GL 路径：`sudo apt install -y libegl-dev libopengl-dev libgles2-mesa-dev ninja-build`；
  或直接使用 `export NVDIFRAST_BACKEND=cuda` 走 CUDA 路径。

* **nvdiffrast JIT 编译失败 / 乱用架构**
  确认 `TORCH_CUDA_ARCH_LIST="12.0+PTX"` 已导出，并清除缓存：`rm -rf ~/.cache/torch_extensions`。

* **显存 OOM**
  降低 `MILO_MESH_RES_SCALE`（如 0.5 → 0.3 → 0.25）、开启三角形分块 `MILO_RAST_TRI_CHUNK`，并使用 `--data_device cpu`。

---

## 结果汇总（本次 Ignatius 流程）

* **训练 (`train.py`)**：完成，输出模型目录 `./output/Ignatius`（含训练状态与日志）。
* **网格提取 (`mesh_extract_sdf.py`)**：得到 **`mesh_learnable_sdf.ply`**，已在 MeshLab 验证可视化。
* **渲染 (`render.py`)**：得到训练/测试视角渲染图像（保存于输出目录的渲染子目录）。
* **指标 (`metrics.py`)**：控制台打印 PSNR/SSIM（并保存到模型目录，文件名以实际实现为准）。

> 如需 Tanks&Temples 评测，可将 `mesh_learnable_sdf.ply` 软链为 `recon.ply`，再按评测脚本执行。

---

## 许可与鸣谢

本仓库为对原始 MILo 项目在 **CUDA 12.8 / RTX 50** 环境下的适配与工程化补充，保留原始项目许可证与归属。感谢原作者与各子模块作者（Tetra-NeRF、nvdiffrast、3D Gaussian Splatting 等）的优秀工作。
