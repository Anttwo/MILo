<div align="center">

<h1>
MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction
</h1>

<font size="5">
SIGGRAPH Asia 2025 - Journal Track (TOG)<br>
</font>

<font size="4">
<a href="https://anttwo.github.io/" style="font-size:100%;">Antoine Guédon*<sup>1</sup></a>&emsp;
<a href="https://www.lix.polytechnique.fr/Labo/Diego.GOMEZ/" style="font-size:100%;">Diego Gomez*<sup>1</sup></a>&emsp;
<a href="https://nissmar.github.io/" style="font-size:100%;">Nissim Maruani<sup>2</sup></a>&emsp;<br>
<a href="https://s2.hk/" style="font-size:100%;">Bingchen Gong<sup>1</sup></a>&emsp;
<a href="https://www-sop.inria.fr/members/George.Drettakis/" style="font-size:100%;">George Drettakis<sup>2</sup></a>&emsp;
<a href="https://www.lix.polytechnique.fr/~maks/" style="font-size:100%;">Maks Ovsjanikov<sup>1</sup></a>&emsp;
</font>
<br>

<font size="4">
<sup>1</sup>Ecole polytechnique, France<br>
<sup>2</sup>Inria, Université Côte d'Azur, France<br>
</font>

<font size="2">
*Both authors contributed equally to the paper.
</font>

| <a href="https://anttwo.github.io/milo">Webpage</a> | <a href="https://arxiv.org/abs/2506.24096">arXiv</a> | <a href="https://www.youtube.com/watch?v=rOBs2yyYaJM">Presentation video</a> | <a href="https://drive.google.com/drive/folders/1Bf7DM2DFtQe4J63bEFLceEycNf4qTcqm?usp=sharing">Data</a> |

![Teaser image](assets/teaser.png)
</div>

## Abstract

_Our method introduces a novel differentiable mesh extraction framework that operates during the optimization of 3D Gaussian Splatting representations. At every training iteration, we differentiably extract a mesh—including both vertex locations and connectivity—only from Gaussian parameters. This enables gradient flow from the mesh to Gaussians, allowing us to promote bidirectional consistency between volumetric (Gaussians) and surface (extracted mesh) representations. This approach guides Gaussians toward configurations better suited for surface reconstruction, resulting in higher quality meshes with significantly fewer vertices. Our framework can be plugged into any Gaussian splatting representation, increasing performance while generating an order of magnitude fewer mesh vertices. MILo makes the reconstructions more practical for downstream applications like physics simulations and animation._

## To-do List

- ⬛ Implement a simple training viewer using the <a href="https://github.com/graphdeco-inria/graphdecoviewer">GraphDeco viewer</a>.
- ⬛ Add the mesh-based rendering evaluation scripts in `./milo/eval/mesh_nvs`.
- ✅ Add low-res and very-low-res training for light output meshes (under 50MB and under 20MB).
- ✅ Add T&T evaluation scripts in `./milo/eval/tnt/`.
- ✅ Add Blender add-on (for mesh-based editing and animation) to the repo.
- ✅ Clean code.
- ✅ Basic refacto.

## License

<details>
<summary>Click here to see content.</summary>

<br>This project builds on existing open-source implementations of various projects cited in the __Acknowledgements__ section.

Specifically, it builds on the original implementation of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting); As a result, parts of this code are licensed under the Gaussian-Splatting License (see `./LICENSE.md`). 

This codebase also builds on various other repositories such as [Nvdiffrast](https://github.com/NVlabs/nvdiffrast); Please refer to the license files of the submodules for more details.
</details>

## 0. Quickstart

<details>
<summary>Click here to see content.</summary>

Please start by creating or downloading a COLMAP dataset, such as <a href="https://drive.google.com/drive/folders/1Bf7DM2DFtQe4J63bEFLceEycNf4qTcqm?usp=sharing">our COLMAP run for the Ignatius scene from the Tanks&Temples dataset</a>. You can move the Ignatius directory to `./milo/data`.

After installing MILo as described in the next section, you can reconstruct a surface mesh from images by going to the `./milo/` directory and running the following commands:

```bash
# Training for an outdoor scene
python train.py -s ./data/Ignatius -m ./output/Ignatius --imp_metric outdoor --rasterizer radegs

# Saves mesh as PLY with vertex colors after training
python mesh_extract_sdf.py -s ./data/Ignatius -m ./output/Ignatius --rasterizer radegs
```
Please change `--imp_metric outdoor` to `--imp_metric indoor` if your scene is indoor.

These commands use the lightest version of our approach, resulting in a small number of Gaussians and a light mesh. You can increase the number of Gaussians by adding `--dense_gaussians`, and improve the robustness to exposure variations with `--decoupled_appearance` as follows:

```bash
# Training with dense gaussians and better appearance model
python train.py -s ./data/Ignatius -m ./output/Ignatius --imp_metric outdoor --rasterizer radegs --dense_gaussians --decoupled_appearance

# Saves mesh as PLY with vertex colors after training
python mesh_extract_sdf.py -s ./data/Ignatius -m ./output/Ignatius --rasterizer radegs
```

Please refer to the following sections for additional details on our training and mesh extraction scripts, including:
- How to use other rasterizers
- How to train MILo with high-resolution meshes
- Various mesh extraction methods
- How to easily integrate MILo's differentiable GS-to-mesh pipeline to your own GS project
</details>


## 1. Installation

<details>
<summary>Click here to see content.</summary>

### Clone this repository.
```bash
git clone https://github.com/Anttwo/MILo.git --recursive
```

### Install dependencies.

Please start by creating an environment:
```bash
conda create -n milo python=3.9
conda activate milo
```

Then, specify your own CUDA paths depending on your CUDA version:
```bash
# You can specify your own cuda path (depending on your CUDA version)
export CPATH=/usr/local/cuda-11.8/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
```

Finally, you can run the following script to install all dependencies, including PyTorch and Gaussian Splatting submodules:
```bash
python install.py
```
By default, the environment will be installed for CUDA 11.8. If using CUDA 12.1, you can provide the argument `--cuda_version 12.1` to `install.py`. **Please note that only CUDA 11.8 has been tested.**

If you encounter problems or if the installation script does not work, please follow the detailed installation steps below.

<details>
<summary>Click here for detailed installation instructions</summary>

```bash
# For CUDA 11.8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 mkl=2023.1.0 -c pytorch -c nvidia

# For CUDA 12.1 (The code has only been tested on CUDA 11.8)
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 mkl=2023.1.0 -c pytorch -c nvidia

pip install -r requirements.txt

# Install submodules for Gaussian Splatting, including different rasterizers, aggressive densification, simplification, and utilities
pip install submodules/diff-gaussian-rasterization_ms
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization_gof
pip install submodules/simple-knn
pip install submodules/fused-ssim

# Delaunay Triangulation from Tetra-Nerf
cd submodules/tetra_triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal

# You can specify your own cuda path (depending on your CUDA version)
export CPATH=/usr/local/cuda-11.8/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH

cmake .
make 
pip install -e .
cd ../../

# Nvdiffrast for efficient mesh rasterization
cd ./submodules/nvdiffrast
pip install .
cd ../../
```

</details>

</details>

## 2. Training with MILo

<details>
<summary>Click here to see content.</summary>

First, go to the MILo folder:
```bash
cd milo
```

Then, to optimize a Gaussian Splatting representation with MILo using a COLMAP dataset, you can run the following command:
```bash
python train.py \
    -s <PATH TO COLMAP DATASET> \
    -m <OUTPUT_DIR> \
    --imp_metric <"indoor" OR "outdoor"> \
    --rasterizer <"radegs" OR "gof">
```
The main arguments are the following:
| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
|  `--imp_metric` | `"indoor"` or `"outdoor"` | Required | Type of scene to optimize. Modifies the importance sampling to better handle indoor or outdoor scenes. |
| `--rasterizer` | `"radegs"` or `"gof"` | `"radegs"` | Rasterization technique used during training. |
| `--dense_gaussians` | flag | disabled | Use more Gaussians during training. When active, only a subset of Gaussians will generate pivots for Delaunay triangulation. When inactive, all Gaussians generate pivots.|

You can use a dense set of Gaussians by adding the argument `--dense_gaussians`:
```bash
python train.py \
    -s <PATH TO COLMAP DATASET> \
    -m <OUTPUT_DIR> \
    --imp_metric <"indoor" OR "outdoor"> \
    --rasterizer <"radegs" OR "gof"> \
    --dense_gaussians \
    --data_device cpu
```

The list of optional arguments is provided below:
| Category | Argument | Values | Default | Description |
|----------|----------|---------|---------|-------------|
| **Performance & Logging** | `--data_device` | `"cpu"` or `"cuda"` | `"cuda"` | Forces data to be loaded on CPU (less GPU memory usage, slightly slower training) |
| | `--log_interval` | integer | - | Log images every N training iterations (e.g., `200`) |
| **Mesh Configuration** | `--mesh_config` | `"default"`, `"highres"`, `"veryhighres"`, `"lowres"`, `"verylowres"` | `"default"` | Config file for mesh resolution and quality |
| **Evaluation & Appearance** | `--eval` | flag | disabled | Performs the usual train/test split for evaluation |
| | `--decoupled_appearance` | flag | disabled | Better handling of exposure variations |
| **Depth-Order Regularization** | `--depth_order` | flag | disabled | Enable depth-order regularization with DepthAnythingV2 |
| | `--depth_order_config` | `"default"` or `"strong"` | `"default"` | Strength of depth regularization |

You can change the config file used during training (useful for ablation runs) by 
specifying `--mesh_config <CONFIG_NAME>`. The different config files are the following:

- **Default config**: The default config file name is `default`. This config results in 
lighter representations and lower resolution meshes, containing around 2M Delaunay vertices 
for the base setting and 5M Delaunay vertices for the `--dense_gaussians` setting.
- **High Res config**: You can use `--mesh_config highres --dense_gaussians` for higher 
resolution meshes. We recommend using this config with `--dense_gaussians`. This config 
results in higher resolution representations, containing up to 9M Delaunay vertices.
- **Very High Res config**: You can use `--mesh_config veryhighres --dense_gaussians` for 
even higher resolution meshes. We recommend using this config with `--dense_gaussians`. 
This config results in even higher resolution representations, containing up to 14M Delaunay 
vertices. This configuration requires more memory for training.
- **Low Res config**: You can use `--mesh_config lowres` for lower resolution meshes (less than 50MB). 
This config results in lower resolution representations, containing up to 500k Delaunay vertices.
You can adjust the number of Gaussians used during training accordingly by decreasing the sampling factor
with `--sampling_factor 0.3`, for instance.
- **Very Low Res config**: You can use `--mesh_config verylowres` for even lower resolution meshes (less than 20MB). 
This config results in even lower resolution representations, containing up to 250k Delaunay vertices.
You can adjust the number of Gaussians used during training accordingly by decreasing the sampling factor
with `--sampling_factor 0.1`, for instance.

Please refer to the <a href="https://depth-anything-v2.github.io/">DepthAnythingV2</a> repo to download the `vitl` checkpoint required for Depth-Order regularization. Then, move the checkpoint file to `./submodules/Depth-Anything-V2/checkpoints/`.

### Example Commands

Basic training for indoor scenes with logging:
```bash
python train.py -s <PATH TO COLMAP DATASET> -m <OUTPUT_DIR> --imp_metric indoor --rasterizer radegs --log_interval 200
```

Dense Gaussians with high resolution in outdoor scenes:
```bash
python train.py -s <PATH TO COLMAP DATASET> -m <OUTPUT_DIR> --imp_metric outdoor --rasterizer radegs --dense_gaussians --mesh_config highres --data_device cpu
```

Full featured training with very high resolution:
```bash
python train.py -s <PATH TO COLMAP DATASET> -m <OUTPUT_DIR> --imp_metric indoor --rasterizer radegs --dense_gaussians --mesh_config veryhighres --decoupled_appearance --log_interval 200 --data_device cpu
```

Very low resolution training in indoor scenes for very light meshes (less than 20MB):
```bash
python train.py -s <PATH TO COLMAP DATASET> -m <OUTPUT_DIR> --imp_metric indoor --rasterizer radegs --sampling_factor 0.1 --mesh_config verylowres
```

Training with depth-order regularization:
```bash
python train.py -s <PATH TO COLMAP DATASET> -m <OUTPUT_DIR> --imp_metric indoor --rasterizer radegs --depth_order --depth_order_config strong --log_interval 200 --data_device cpu
```

</details>

## 3. Extracting a Mesh after Optimization

<details>
<summary>Click here to see content.</summary>

First go to `./milo/`.

### 3.1. Use learned SDF values

You can then use the following command:
```bash
python mesh_extract_sdf.py \
    -s <PATH TO COLMAP DATASET> \
    -m <MODEL DIR> \
    --rasterizer <"radegs" OR "gof">
```
This script will further refine the SDF values for a short period of time (1000 iterations by default) with frozen Gaussians, then save the mesh as a PLY file with vertex colors. The mesh will be located at `<MODEL_DIR>/mesh_learnable_sdf.ply`.

**WARNING:** Make sure you use the same mesh config file as the one used during training. You can change the config file by specifying `--config <CONFIG_NAME>`. The default config file name is `default`, but you can switch to `highres`, `veryhighres`, `lowres` or `verylowres`.

You can use the usual train/test split by adding the argument `--eval`. 

### 3.2. Use Integrated Opacity Field or scalable TSDF

To extract a mesh using the integrated opacity field as defined by the Gaussians in GOF and RaDe-GS, you can run the following command:
```bash
python mesh_extract_integration.py \
    -s <PATH TO COLMAP DATASET> \
    -m <MODEL DIR>
```
You can use the argument `--rasterizer <radegs OR gof>` to change the rasterization technique for computing the opacity field. Default is `gof`. We recommend using GOF in this context (even if RaDe-GS was used during training), as the opacity computation from GOF is more precise and will produce less surface erosion.

You can also use the argument `--sdf_mode <"integration" OR "depth_fusion">` to modify the SDF computation strategy. Default mode is `integration`, which uses the integrated opacity field. Please note that `depth_fusion` is not traditional TSDF performed on a regular grid, but our more efficient depth fusion strategy relying on the same Gaussian pivots as the ones used for `integration`.

If using `integration`, you can modify the isosurface value with the argument `--isosurface_value <value>`. The default value is 0.5.
```bash
python mesh_extract_integration.py \
    -s <PATH TO COLMAP DATASET> \
    -m <MODEL DIR> \
    --rasterizer gof \
    --sdf_mode integration \
    --isosurface_value 0.5
```

If using `depth_fusion`, you can modify the truncation margin with the argument `--trunc_margin <value>`. If not provided, the value is automatically computed depending on the scale of the scene. We recommend not changing this value.
```bash
python mesh_extract_integration.py \
    -s <PATH TO COLMAP DATASET> \
    -m <MODEL DIR> \
    --rasterizer gof \
    --sdf_mode depth_fusion \
    --trunc_margin 0.005
```

The mesh will be saved at either `<MODEL_DIR>/mesh_integration_sdf.ply` or `<MODEL_DIR>/mesh_depth_fusion_sdf.ply` depending on the SDF computation method.

</details>

## 4. Using our differentiable Gaussians-to-Mesh pipeline in your own 3DGS project

<details>
<summary>Click here to see content.</summary>
<br>

In `milo.functional`, we provide straightforward functions to leverage our differentiable *Gaussians-to-Mesh pipeline* in your own 3DGS projects.

These functions only require Gaussian parameters as inputs (`means`, `scales`, `rotations`, `opacities`) and can extract a mesh from these parameters in a differentiable manner, allowing for **performing differentiable operations on the surface mesh and backpropating gradients directly to the Gaussians**.

We only assume that your own `Camera` class has the same structure as the class from the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) implementation.

Specifically, we propose the following functions:
- `sample_gaussians_on_surface`: This function samples Gaussians that are the most likely to be located on the surface of the scene. For more efficiency, we propose using only these Gaussians for generating pivots and applying Delaunay triangulation.
- `extract_gaussian_pivots`: This differentiable function builds pivots from the parameters of the sampled Gaussians. In practice, there is no need to explicitely call this function, as our other functions can recompute pivots on the fly. However, you might want to perform special treatment on the pivots.
- `compute_initial_sdf_values`: This function estimates initial truncated SDF values for any set of Gaussian pivots by performing depth-fusion over the provided viewpoints. You can directly provide the gaussian parameters to this function, in which case pivots will be computed on the fly. In the paper, we propose to learn optimal SDF values by maximizing the consistency between volumetric GS renderings and surface mesh renderings; We use this function to initialize the SDF values.
- `compute_delaunay_triangulation`: This function computes the Delaunay triangulation for a set of sampled Gaussians pivots. You can provide either pivots as inputs, or directly the parameters of the Gaussians (means, scales, rotations...), in which case the pivots will be recomputed on the fly. This function should not be applied at every training iteration as it is very slow, and the Delaunay graph does not change that much during training.
- `extract_mesh`: This differentiable function extracts the mesh from the Gaussian parameters, given a Delaunay triangulation and SDF values for the Gaussian pivots.

We also propose additional functions such as `frustum_cull_mesh` which culls mesh vertices based on the view frustum of an input camera.

We provide an example of how to use these functions below, using our codebase or any codebase following the same template as the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) implementation.

```python
from functional import (
    sample_gaussians_on_surface,
    extract_gaussian_pivots,
    compute_initial_sdf_values,
    compute_delaunay_triangulation,
    extract_mesh,
    frustum_cull_mesh,
)

# Load or initialize a 3DGS-like model and training cameras
gaussians = ...
train_cameras = ...

# Define a simple wrapper for your Gaussian Splatting rendering function, 
# following this template. It will be used only for initializing SDF values.
# The wrapper should accept just a camera as input, and return a dictionary 
# with "render" and "depth" keys.
from gaussian_renderer.radegs import render_radegs
pipe = ...
background = torch.tensor([0., 0., 0.], device="cuda")
def render_func(view):
    render_pkg = render_radegs(
        viewpoint_camera=view, 
        pc=gaussians, 
        pipe=pipe, 
        bg_color=background, 
        kernel_size=0.0, 
        scaling_modifier = 1.0, 
        require_coord=False, 
        require_depth=True
    )
    return {
        "render": render_pkg["render"],
        "depth": render_pkg["median_depth"],
    }

# Only the parameters of the Gaussians are needed for extracting the mesh.
means = gaussians.get_xyz
scales = gaussians.get_scaling
rotations = gaussians.get_rotation
opacities = gaussians.get_opacity

# Sample Gaussians on the surface.
# Should be performed only once, or just once in a while.
# In this example, we sample at most 600_000 Gaussians.
surface_gaussians_idx = sample_gaussians_on_surface(
    views=train_cameras,
    means=means,
    scales=scales,
    rotations=rotations,
    opacities=opacities,
    n_max_samples=600_000,
    scene_type='indoor',
)

# Compute initial SDF values for pivots. Should be performed only once.
# In the paper, we propose to learn optimal SDF values by maximizing the 
# consistency between volumetric renderings and surface mesh renderings.
initial_pivots_sdf = compute_initial_sdf_values(
    views=train_cameras,
    render_func=render_func,
    means=means,
    scales=scales,
    rotations=rotations,
    gaussian_idx=surface_gaussians_idx,
)

# Compute Delaunay Triangulation.
# Can be performed once in a while.
delaunay_tets = compute_delaunay_triangulation(
    means=means,
    scales=scales,
    rotations=rotations,
    gaussian_idx=surface_gaussians_idx,
)

# Differentiably extract a mesh from Gaussian parameters, including initial 
# or updated SDF values for the Gaussian pivots.
# This function is differentiable with respect to the parameters of the Gaussians, 
# as well as the SDF values. Can be performed at every training iteration.
mesh = extract_mesh(
    delaunay_tets=delaunay_tets,
    pivots_sdf=initial_pivots_sdf,
    means=means,
    scales=scales,
    rotations=rotations,
    gaussian_idx=surface_gaussians_idx,
)

# You can now apply any differentiable operation on the extracted mesh, 
# and backpropagate gradients back to the Gaussians!
# In the paper, we propose to use differentiable mesh rendering.
from scene.mesh import MeshRasterizer, MeshRenderer
renderer = MeshRenderer(MeshRasterizer(cameras=train_cameras))

# We cull the mesh based on the view frustum for more efficiency
i_view = np.random.randint(0, len(train_cameras))
mesh_render_pkg = renderer(
    frustum_cull_mesh(mesh, train_cameras[i_view]), 
    cam_idx=i_view, 
    return_depth=True, return_normals=True
)
mesh_depth = mesh_render_pkg["depth"]
mesh_normals = mesh_render_pkg["normals"]
```

</details>

## 5. Creating a COLMAP dataset with your own images

<details>
<summary><span>Click here to see content.</span></summary>
<br>

### 5.1. Estimate camera poses with COLMAP

Please first install a recent version of COLMAP (ideally CUDA-powered) and make sure to put the images you want to use in a directory `<location>/input`. Then, run the script `milo/convert.py` from the original Gaussian splatting implementation to compute the camera poses for the images using COLMAP. Please refer to the original <a href="https://github.com/graphdeco-inria/gaussian-splatting">3D Gaussian Splatting repository</a> for more details.

```shell
python milo/convert.py -s <location>
```

Sometimes COLMAP fails to reconstruct all images into the same model and hence produces multiple sub-models. The smaller sub-models generally contain only a few images. However, by default, the script `convert.py` will apply Image Undistortion only on the first sub-model, which may contain only a few images.

If this is the case, a simple solution is to keep only the largest sub-model and discard the others. To do this, open the source directory containing your input images, then open the sub-directory `<Source_directory>/distorted/sparse/`. You should see several sub-directories named `0/`, `1/`, etc., each containing a sub-model. Remove all sub-directories except the one containing the largest files, and rename it to `0/`. Then, run the script `convert.py` one more time but skip the matching process:

```shell
python milo/convert.py -s <location> --skip_matching
```

_Note: If the sub-models have common registered images, they could be merged into a single model as post-processing step using COLMAP; However, merging sub-models requires to run another global bundle adjustment after the merge, which can be time consuming._

### 5.2. Estimate camera poses with VGGT

Coming soon.

</details>


## 6. Mesh-Based Editing and Animation with the MILo Blender Addon

<details>
<summary>Click here to see content.</summary>
<br>

While MILo provides a differentiable solution for extracting meshes from 3DGS representations, it also implicitly encourages Gaussians to align with the surface of the mesh. As a result, any modification made to the mesh can be easily propagated to the Gaussians, making the reconstructed mesh an excellent proxy for editing and animating the 3DGS representation.

Similarly to previous works <a href="https://anttwo.github.io/sugar/">SuGaR</a> and <a href="https://anttwo.github.io/frosting/">Gaussian Frosting</a>, we provide a Blender addon allowing to combine, edit and animate 3DGS representations just by manipulating meshes reconstructed with MILo in Blender.

### 6.1. Installing the addon

1. Please start by installing `torch_geometric` and `torch_cluster` in your `milo` conda environment:
```shell
pip install torch_geometric
pip install torch_cluster
```

2. Then, install <a href="https://www.blender.org/download/">Blender</a> (version 4.0.2 is recommended but not mandatory).

3. Open Blender, and go to `Edit` > `Preferences` > `Add-ons` > `Install`, and select the file `milo_addon.py` located in `./milo/blender/`.<br>

You have now installed the MILo addon for Blender!

### 6.2. Usage

This Blender addon is almost identical as the <a href="https://github.com/Anttwo/sugar_frosting_blender_addon">SuGaR x Frosting Blender addon</a>. You can refer to this previous repo for more details and illustrations. To combine, edit or animate scene with the addon, please follow the steps below:

1. Please start by training Gaussians with MILo and extracting a mesh, as described in the Quickstart section.

2. Open a new scene in Blender, and go to the `Render` tab in the Properties. You should see a panel named `Add MILo mesh`. The panel is not necessary at the top of the tab, so you may need to scroll down to find it.

3. **(a) Select a mesh.** Enter the path to the final mesh extracted with MILo in the `Path to mesh PLY` field. You can also click on the folder icon to select the file. The mesh should be located at `<path to model directory>/mesh_learnable_sdf.ply`.<br><br>
**(b) Select a checkpoint.** Similarly, enter the path to the final checkpoint of the optimization in the `Path to 3DGS PLY` field. You can also click on the folder icon to select the file. The checkpoint should be located at `<path to model directory>/point_cloud/iteration_18000/point_cloud.ply`.<br><br>
**(c) Load the mesh.** Finally, click on `Add mesh` to load the mesh in Blender. Feel free to rotate the mesh and change the shading mode to better visualize the mesh and its colors. 

4. **Now, feel free to edit your mesh using Blender!** 
<br>You can segment it into different pieces, sculpt it, rig it, animate it using a parent armature, *etc*. You can also add other MILo meshes to the scene, and combine elements from different scenes. <br>
Feel free to set a camera in the scene and prepare an animation: You can animate the camera, the mesh, *etc*.<br>
Please avoid using `Apply Location`, `Apply Rotation`, or `Apply Scale` on the edited mesh, as we are still unsure how it will affect the correspondence between the mesh and the optimized checkpoint.

5. Once you're done with your editing, you can prepare a rendering package ready to be rendered with Gaussians. To do so, go to the `Render` tab in the Properties again, and select the `./milo/` directory in the `Path to MILo directory` field.<br> 
Finally, click on `Render Image` or `Render Animation` to save a rendering package for the scene. <br><br>
`Render Image` will render a single image of the scene, with the current camera position and mesh editions/poses.<br><br>
`Render Animation` will render a full animation of the scene, from the first frame to the last frame you set in the Blender Timeline.
<br><br>
The package should be saved as a `JSON` file and located in `./milo/blender/packages/`.

7. Finally, you can render the package with Gaussian Splatting. You just need to go to `./milo/` and run the following command:
```shell
python render_blender_scene.py \
    -p <path to the package json file> \
    --rasterizer <"radegs" or "gof">.
```

By default, renderings are saved in `./milo/blender/renders/<name of the scene>/`. However, you can change the output directory by adding `-o <path to output directory>`.

Please check the documentation of the `render_blender_scene.py` scripts for more information on the additional arguments.
If you get artifacts in the rendering, you can try to play with the various following hyperparameters: `binding_mode`, `filter_big_gaussians_with_th`, `clamp_big_gaussians_with_th`, and `filter_distant_gaussians_with_th`.

</details>

## 7. Evaluation

<details>
<summary>Click here to see content.</summary>
<br>

For evaluation, please start by downloading [our COLMAP runs for the Tanks and Temples dataset](https://drive.google.com/drive/folders/1Bf7DM2DFtQe4J63bEFLceEycNf4qTcqm?usp=sharing), and make sure to move all COLMAP scene directories (Barn, Caterpillar, _etc._) inside the same directory. 

Then, please download ground truth point cloud, camera poses, alignments and cropfiles from [Tanks and Temples dataset](https://www.tanksandtemples.org/download/). The ground truth dataset should be organized as:
```
GT_TNT_dataset
│
└─── Barn
│   │
|   └─── Barn.json
│   │
|   └─── Barn.ply
│   │
|   └─── Barn_COLMAP_SfM.log
│   │
|   └─── Barn_trans.txt
│ 
└─── Caterpillar
│   │
......
```

We follow the exact same pipeline as GOF and RaDe-GS for evaluating MILo on T&T. Please go to `./milo/` and run the following script to run the full training and evaluation pipeline on all scenes:

```bash
python scripts/evaluate_tnt.py \
    --data_dir <path to directory containing TNT COLMAP datasets> \
    --gt_dir <path to the GT TNT directory> \
    --output_dir <path to output directory> \
    --rasterizer <"radegs" or "gof"> \
    --mesh_config <"default" or "highres" or "veryhighres">
```
You can add `--dense_gaussians` for using more Gaussians during training. Please note that `--dense_gaussians` will be automatically set to `True` if using `--mesh_config highres` or `--mesh_config veryhighres`.

For evaluating only a single scene, you can run the following commands:

```bash
# Training (you can add --dense_gaussians for higher performance)
python train.py \
    -s <path to preprocessed TNT dataset> \
    -m <output folder> \
    --imp_metric <"indoor" or "outdoor"> \
    --rasterizer <"radegs" or "gof"> \
    --mesh_config <"default" or "highres" or "veryhighres"> \
    --eval \
    --decoupled_appearance \
    --data_device cpu

# Mesh extraction
python mesh_extract_sdf.py \
    -s <path to preprocessed TNT dataset> \
    -m <output folder> \
    --rasterizer <"radegs" or "gof"> \
    --config <"default" or "highres" or "veryhighres"> \
    --eval \
    --data_device cpu

# Evaluation
python eval/tnt/run.py \
    --dataset-dir <path to GT TNT dataset> \
    --traj-path <path to preprocessed TNT COLMAP_SfM.log file> \
    --ply-path <output folder>/recon.ply
```

### Novel View Synthesis
After training MILo on a scene with test/train split by using the argument `--eval`, you can evaluate the performance of the Novel View Synthesis by running the scripts below:

```bash
python render.py \
    -m <path to pre-trained model> \
    -s <path to dataset> \
    -- rasterizer <"radegs" or "gof">

python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

</details>

## 8. Acknowledgements

We build this project based on [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Mini-Splatting2](https://github.com/fatPeter/mini-splatting2).

We propose to use rasterization techniques from [RaDe-GS](https://baowenz.github.io/radegs/) and [GOF](https://github.com/autonomousvision/gaussian-opacity-fields/tree/main).

The latter incorporate the filters proposed in [Mip-Splatting](https://github.com/autonomousvision/mip-splatting), the loss functions of [2D GS](https://github.com/hbb1/2d-gaussian-splatting) and its preprocessed DTU dataset.

We use [Nvdiffrast](https://github.com/NVlabs/nvdiffrast) for differentiable triangle rasterization, and [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) for computing our optional depth-order regularization loss relying on monocular depth estimation.

The evaluation scripts for the Tanks and Temples dataset are sourced from [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation).

We thank the authors of all the above projects for their great works.
