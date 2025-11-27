# MILo_rtx50 — CUDA 12.8 / RTX 50 Series Local Compilation and Execution Guide (Ubuntu 24.04 + uv + PyTorch 2.7.1+cu128)

> This README documents our **local compilation adaptation, key modifications, and reproducible experimental steps** for the forked project [Anttwo/MILo](https://github.com/Anttwo/MILo) in the **RTX 50 series + CUDA 12.8** environment.
> Goal: Complete submodule compilation, training, mesh extraction, rendering, and evaluation using **uv + venv** without Conda.

---

## Environment

- OS: Ubuntu 24.04
- GPU: RTX 50 Series (Blackwell)
- CUDA Toolkit: 12.8 (NVCC `/usr/local/cuda-12.8/bin/nvcc`)
- Python: 3.12.3 (venv management, package management with **uv**)
- PyTorch: **2.7.1+cu128** (official binary, **C++11 ABI=1**)
- C/C++: GCC 13.3
- CMake: System version (apt)
- Important environment variables (commonly used during training/extraction/rendering):
  ```bash
  export NVDIFRAST_BACKEND=cuda
  export TORCH_CUDA_ARCH_LIST="12.0+PTX"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32,garbage_collection_threshold:0.6"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  # Mesh regularization grid resolution scaling, smaller value saves VRAM
  export MILO_MESH_RES_SCALE=0.3
  # (Optional) Triangle chunk size to mitigate nvdiffrast CUDA backend VRAM peaks
  export MILO_RAST_TRI_CHUNK=150000
  ```

---

## Submodules Installation

> We have verified these can be successfully compiled/installed with CUDA 12.8 + PyTorch 2.7.1.

### 1) Install Gaussian Splatting Submodules (via **pip**)

```bash
pip install submodules/diff-gaussian-rasterization_ms
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization_gof
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

> Note: `nvdiffrast` uses **JIT compilation** (triggered at runtime by PyTorch cpp_extension).
> If choosing **OpenGL(GL) backend**, system headers are required: `sudo apt install -y libegl-dev libopengl-dev libgles2-mesa-dev ninja-build`.
> We switched to **CUDA backend** for simplicity: `export NVDIFRAST_BACKEND=cuda` (no EGL headers needed).

### 2) Install System Dependencies for `tetra_triangulation` (Delaunay Triangulation)

The original project uses **conda** to install system-level C/C++ dependencies (cmake/gmp/cgal). Since we use **uv** for Python packages only, we need to install these C/C++ libraries via **apt** (system package manager):

```bash
# Install C/C++ dependencies via apt (Ubuntu 24.04)
sudo apt update
sudo apt install -y \
  build-essential \
  cmake ninja-build \
  libgmp-dev libmpfr-dev libcgal-dev \
  libboost-all-dev

# (Optional) May be needed:
# sudo apt install -y libeigen3-dev
```

**Notes:**
- `libcgal-dev` provides CGAL headers (header-only on Ubuntu 24.04)
- `libgmp-dev` and `libmpfr-dev` are numerical backends for CGAL
- **uv only manages Python packages**; C/C++ dependencies must be installed via system package managers (apt/brew/pacman)
- For **macOS**: `brew install cmake cgal gmp mpfr boost`
- For **Arch Linux**: `sudo pacman -S cgal gmp mpfr boost cmake base-devel`

### 3) Compile `tetra_triangulation` with ABI Alignment

> **Important:** This module requires ABI alignment with PyTorch 2.7.1 (C++11 ABI=1). We use a header file approach to enforce this.

**a) Create ABI enforcement header:**

Create `submodules/tetra_triangulation/src/force_abi.h`:
```cpp
#pragma once
// Force new ABI before any STL headers
#if defined(_GLIBCXX_USE_CXX11_ABI)
#  undef _GLIBCXX_USE_CXX11_ABI
#endif
#define _GLIBCXX_USE_CXX11_ABI 1
```

**b) Modify source files:**

Add `#include "force_abi.h"` as the **first line** of:
- `submodules/tetra_triangulation/src/py_binding.cpp`
- `submodules/tetra_triangulation/src/triangulation.cpp`

**c) Build and install:**

```bash
cd submodules/tetra_triangulation
rm -rf build CMakeCache.txt CMakeFiles tetranerf/utils/extension/tetranerf_cpp_extension*.so

# Point to current PyTorch's CMake prefix/dynamic library path
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

# Install (optional, convenient for editable reference)
uv pip install -e .
cd ../../
```

> **Note:** For troubleshooting ABI issues, see **Key Issue 1** section below.

---

## Key Issue 1: `tetra_triangulation` ABI Mismatch (Resolved)

**Symptom**
Running `from tetranerf.utils import extension as ext` throws error:

```
undefined symbol: _ZN3c106detail14torchCheckFailEPKcS2_jRKSs
```

The trailing `RKSs` indicates **old ABI (_GLIBCXX_USE_CXX11_ABI=0)**, while our PyTorch 2.7.1 uses **new ABI (=1)**.

**Fix**
Add a header file to force ABI in `submodules/tetra_triangulation` to **stably lock ABI=1**:

* Create file: `src/force_abi.h`

  ```cpp
  #pragma once
  // Force new ABI before any STL headers
  #if defined(_GLIBCXX_USE_CXX11_ABI)
  #  undef _GLIBCXX_USE_CXX11_ABI
  #endif
  #define _GLIBCXX_USE_CXX11_ABI 1
  ```

* Modify: Add `#include "force_abi.h"` as the **first line** of `src/py_binding.cpp` and `src/triangulation.cpp`

  ```cpp
  #include "force_abi.h"
  ```

> **Note:** This header file approach is sufficient to enforce ABI=1. No additional CMakeLists.txt modifications are needed.

**Build Commands (in-source, outputs to package path)**

```bash
cd submodules/tetra_triangulation
rm -rf build CMakeCache.txt CMakeFiles tetranerf/utils/extension/tetranerf_cpp_extension*.so

# Point to current PyTorch's CMake prefix/dynamic library path
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

# Install (optional, convenient for editable reference)
uv pip install -e .
```

---

## Key Issue 2: nvdiffrast GL Backend Missing `EGL/egl.h` (Bypassed)

* Option A: `sudo apt install -y libegl-dev libopengl-dev libgles2-mesa-dev` and continue with GL.
* **Option B (We adopted)**: Switch to **CUDA backend**: `export NVDIFRAST_BACKEND=cuda`, no EGL header dependency.

---

## Key Issue 3: nvdiffrast CUDA Backend VRAM OOM (Resolved)

**Symptom**
`cudaMalloc(&m_gpuPtr, bytes)` OOM (error: 2), especially during Mesh regularization phase.

**Fix (Two Points)**

1. Replace `nvdiff_rasterization` implementation in `milo/scene/mesh.py`:

   * Support **triangle chunking** (env variable `MILO_RAST_TRI_CHUNK` specifies chunk size)
   * **Fix CUDA backend `ranges` must be on CPU** (`dr.rasterize(..., ranges=<CPU tensor>)`)

   <details>
   <summary>Modified function (click to expand)</summary>

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
       Replacement version equivalent to original function, supports triangle chunking (env: MILO_RAST_TRI_CHUNK),
       and fixes: nvdiffrast CUDA backend's `ranges` must be on CPU.
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
           # ranges must be on CPU
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

2. Reduce memory peak at runtime:

   * `MILO_MESH_RES_SCALE=0.3` (mesh regularization resolution scaling)
   * `MILO_RAST_TRI_CHUNK=150000` (triangle chunk size)
   * `--data_device cpu` (cameras/data on CPU)

---

## Reproduction Steps (Ignatius)

> Data path: `./data/Ignatius`
> Output directory: `./output/Ignatius`

### 1) Training

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

**Output** (located in `./output/Ignatius`)

* Trained scene (Gaussians + learnable SDF and mesh regularization state, etc.)
* Logs and intermediate files (as configured by script, console prints training progress)

### 2) Mesh Extraction (SDF)

```bash
python mesh_extract_sdf.py \
  -s ./data/Ignatius \
  -m ./output/Ignatius \
  --rasterizer gof \
  --config verylowres \
  --data_device cpu
```

**Output**

* **`./output/Ignatius/mesh_learnable_sdf.ply`** (confirmed to open normally in MeshLab)

### 3) Rendering

```bash
python render.py \
  -m ./output/Ignatius \
  -s ./data/Ignatius \
  --rasterizer gof \
  --eval
```

**Output**

* Rendered images (train/test views), saved to the rendering subdirectory in the model output directory (as indicated by script output)

### 4) Image Metrics

```bash
python metrics.py -m ./output/Ignatius
```

**Output**

* Console output of PSNR/SSIM (and corresponding files saved by repo script, located in model directory; based on actual implementation)

### 5) PLY Format Conversion (Optional)

The extracted PLY mesh can be converted to other common 3D formats (OBJ/GLB) using the `clean_convert_mesh.py` script for use in various 3D software. The script also provides optional mesh cleaning functionality.

**Install Additional Dependencies**
```bash
pip install pymeshlab trimesh plyfile
```

**Basic Usage**
```bash
# Basic conversion (outputs PLY, OBJ, GLB)
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply

# Convert and simplify to 300k triangles
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --simplify 300000

# Clean small components during conversion (default 0.02 = remove components with diameter < 2% bbox diagonal)
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --keep-components 0.02

# Output only specific formats
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --no-glb  # Skip GLB
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply --no-obj  # Skip OBJ

# Specify output directory and filename
python clean_convert_mesh.py --in ./output/Ignatius/mesh_learnable_sdf.ply \
  --out-dir ./output/Ignatius/converted \
  --stem mesh_final
```

**Main Features**
- **Format Conversion**: Convert PLY to OBJ and GLB formats (suitable for different 3D software and Web display)
- **Optional Cleaning**: Remove duplicate vertices/faces, fix non-manifold edges, remove small floating components
- **Optional Simplification**: Shape-preserving simplification based on Quadric decimation

**Output** (saved in input file directory by default)
* `mesh_clean.ply` - Converted PLY mesh (with vertex colors)
* `mesh_clean.obj` - OBJ format (Note: OBJ doesn't support vertex colors)
* `mesh_clean.glb` - GLB format (suitable for Web display and import into Blender/Unity etc.)

---

## Our Modifications (Relative to Upstream)

1. **`submodules/tetra_triangulation`**

   * Added `src/force_abi.h`, and `#include "force_abi.h"` at the first line of `src/py_binding.cpp` and `src/triangulation.cpp`: **Force use of C++11 new ABI (=1)**

2. **`milo/scene/mesh.py`**

   * Replaced `nvdiff_rasterization`:

     * Support **MILO_RAST_TRI_CHUNK** triangle chunking
     * **Fixed CUDA backend `ranges` must be CPU Tensor**
     * Other behavior remains consistent with original function

3. **Runtime Configuration**

   * Default to nvdiffrast **CUDA** backend (`NVDIFRAST_BACKEND=cuda`), avoiding EGL dependency
   * Specify `TORCH_CUDA_ARCH_LIST="12.0+PTX"` for Blackwell
   * Reduce peak VRAM: `MILO_MESH_RES_SCALE=0.3`, `MILO_RAST_TRI_CHUNK=150000`, `--data_device cpu`

---

## Common Issues and Troubleshooting

* **`undefined symbol: ... torchCheckFail ... RKSs`**
  This is an ABI=0 symbol; please recompile `tetra_triangulation` with the above patch.

* **`fatal error: EGL/egl.h: No such file or directory`**
  If insisting on GL path: `sudo apt install -y libegl-dev libopengl-dev libgles2-mesa-dev ninja-build`;
  Or directly use `export NVDIFRAST_BACKEND=cuda` for CUDA path.

* **nvdiffrast JIT compilation failure / wrong architecture**
  Confirm `TORCH_CUDA_ARCH_LIST="12.0+PTX"` is exported, and clear cache: `rm -rf ~/.cache/torch_extensions`.

* **VRAM OOM**
  Reduce `MILO_MESH_RES_SCALE` (e.g., 0.5 → 0.3 → 0.25), enable triangle chunking `MILO_RAST_TRI_CHUNK`, and use `--data_device cpu`.

---

## Results Summary (This Ignatius Pipeline)

* **Training (`train.py`)**: Completed, output model directory `./output/Ignatius` (contains training state and logs).
* **Mesh Extraction (`mesh_extract_sdf.py`)**: Obtained **`mesh_learnable_sdf.ply`**, verified visualization in MeshLab.
* **Rendering (`render.py`)**: Obtained rendered images from train/test views (saved in rendering subdirectory of output directory).
* **Metrics (`metrics.py`)**: Console prints PSNR/SSIM (and saves to model directory, filename based on actual implementation).

> For Tanks&Temples evaluation, you can symlink `mesh_learnable_sdf.ply` as `recon.ply`, then run evaluation scripts.

---

## License and Acknowledgments

This repository is an adaptation and engineering supplement to the original MILo project in the **CUDA 12.8 / RTX 50** environment, retaining the original project license and attribution. Thanks to the original authors and all submodule authors (Tetra-NeRF, nvdiffrast, 3D Gaussian Splatting, etc.) for their excellent work.