from typing import List, Union, Tuple, Optional
import torch
import nvdiffrast.torch as dr
from scene.cameras import Camera
from utils.geometry_utils import transform_points_world_to_view
import os
import nvdiffrast.torch as dr


def nvdiff_rasterization_old(
    camera,
    image_height:int, 
    image_width:int,
    verts:torch.Tensor, 
    faces:torch.Tensor,
    return_indices_only:bool=False,
    glctx=None,
    return_rast_out:bool=False,
    return_positions:bool=False,
):
    device = verts.device
        
    # Get full projection matrix
    camera_mtx = camera.full_proj_transform
    
    # Convert to homogeneous coordinates
    pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device)], axis=1)
    
    # Transform points to NDC/clip space
    pos = torch.matmul(pos, camera_mtx)[None]
    
    # Rasterize with NVDiffRast
    # TODO: WARNING: pix_to_face is not in the correct range [-1, F-1] but in [0, F],
    # With 0 indicating that no triangle was hit.
    # So we need to subtract 1.
    rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[image_height, image_width])
    bary_coords, zbuf, pix_to_face = rast_out[..., :2], rast_out[..., 2], rast_out[..., 3].int() - 1
    
    if return_indices_only:
        return pix_to_face
    
    _output = (bary_coords, zbuf, pix_to_face)
    if return_rast_out:
        _output = _output + (rast_out,)
    if return_positions:
        _output = _output + (pos,)
    return _output

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

    # 1) 裁剪空间坐标（与原版一致）
    cam_mtx = camera.full_proj_transform
    pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device, dtype=dtype)], dim=1)
    pos = torch.matmul(pos, cam_mtx)[None]  # [1,V,4]

    # 2) 准备 tri / 形参
    faces = faces.to(torch.int32).contiguous()
    # 与 pos 同设备更稳（nvdiffrast 在很多场景都支持 tri 在 GPU/CPU，但统一更省心）
    faces_dev = faces.to(pos.device)

    H, W = int(image_height), int(image_width)
    chunk = int(os.getenv("MILO_RAST_TRI_CHUNK", "0") or "0")
    use_chunking = chunk > 0 and faces.shape[0] > chunk

    # 快速路径：不分块，完全等价原实现
    if not use_chunking:
        rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces_dev, resolution=[H, W])
        bary_coords = rast_out[..., :2]
        zbuf = rast_out[..., 2]
        pix_to_face = rast_out[..., 3].to(torch.int32) - 1  # 未命中 => -1
        if return_indices_only:
            return pix_to_face
        _out = (bary_coords, zbuf, pix_to_face)
        if return_rast_out:
            _out += (rast_out,)
        if return_positions:
            _out += (pos,)
        return _out

    # 分块路径：ranges 必须在 CPU！
    z_ndc = (pos[..., 2:3] / (pos[..., 3:4] + 1e-20)).contiguous()  # [1,V,1]

    best_rast = None
    best_depth = None
    n_faces = int(faces.shape[0])
    start = 0

    def _normalize_tri_id(rast_chunk, start_idx, count_idx):
        tri_raw = rast_chunk[..., 3:4].to(torch.int64)  # >0 => 命中
        if tri_raw.numel() == 0:
            return rast_chunk[..., 3:4]
        maxid = int(tri_raw.max().item())
        if maxid == 0:
            return rast_chunk[..., 3:4]
        # 如果是局部编号（1..count），平移为全局（1..n_faces）
        if maxid <= count_idx:
            tri_adj = torch.where(tri_raw > 0, tri_raw + start_idx, tri_raw)
        else:
            tri_adj = tri_raw
        return tri_adj.to(rast_chunk.dtype)

    while start < n_faces:
        count = min(chunk, n_faces - start)
        # ！！！关键修正：ranges 在 CPU 上！！！
        ranges_cpu = torch.tensor([[start, count]], device="cpu", dtype=torch.int32)

        rast_chunk, _ = dr.rasterize(
            glctx, pos=pos, tri=faces_dev, resolution=[H, W], ranges=ranges_cpu
        )

        depth_chunk, _ = dr.interpolate(z_ndc, rast_chunk, faces_dev)  # [1,H,W,1]
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
    pix_to_face = rast_out[..., 3].to(torch.int32) - 1  # 未命中 => -1

    if return_indices_only:
        return pix_to_face

    _output = (bary_coords, zbuf, pix_to_face)
    if return_rast_out:
        _output += (rast_out,)
    if return_positions:
        _output += (pos,)
    return _output




class Meshes(torch.nn.Module):
    """
    Meshes class for storing meshes parameters.
    """
    def __init__(
        self, 
        verts:torch.Tensor, 
        faces:torch.Tensor, 
        verts_colors:torch.Tensor=None
    ):
        super().__init__()
        assert verts_colors is None or verts_colors.shape[0] == verts.shape[0]
        self.verts = verts
        self.faces = faces.to(torch.int32)
        self.verts_colors = verts_colors
        
    @property
    def face_normals(self):
        faces_verts = self.verts[self.faces]
        faces_verts_normals = torch.cross(
            faces_verts[:,1] - faces_verts[:,0], 
            faces_verts[:,2] - faces_verts[:,0], 
            dim=-1
        )
        faces_verts_normals = torch.nn.functional.normalize(faces_verts_normals, dim=-1)
        return faces_verts_normals
    
    @property
    def vertex_normals(self):
        raise NotImplementedError("Vertex normals are not implemented yet")
    
    def submesh(
        self, 
        vert_idx:Optional[torch.Tensor]=None, 
        face_idx:Optional[torch.Tensor]=None,
        vert_mask:Optional[torch.Tensor]=None,
        face_mask:Optional[torch.Tensor]=None,
    ):
        assert (
            (vert_idx is not None) or (vert_mask is not None) 
            or
            (face_idx is not None) or (face_mask is not None)
        ), "Either vert_idx, vert_mask, face_idx, or face_mask must be provided"

        if (vert_idx is not None) or (vert_mask is not None):
            if vert_mask is None:
                vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
                vert_mask[vert_idx] = True
            face_mask = vert_mask[self.faces].all(dim=1)

        elif (face_idx is not None) or (face_mask is not None):
            if face_mask is None:
                face_mask = torch.zeros(self.faces.shape[0], dtype=torch.bool, device=self.verts.device)
                face_mask[face_idx] = True
            vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
            vert_mask[self.faces[face_mask]] = True
        
        old_vert_idx_to_new_vert_idx = torch.zeros(self.verts.shape[0], dtype=self.faces.dtype, device=self.verts.device)
        old_vert_idx_to_new_vert_idx[vert_mask] = torch.arange(vert_mask.sum(), dtype=self.faces.dtype, device=self.verts.device)
        
        new_verts = self.verts[vert_mask]
        new_verts_colors = None if self.verts_colors is None else self.verts_colors[vert_mask]
        new_faces = old_vert_idx_to_new_vert_idx[self.faces][face_mask]
        
        return Meshes(verts=new_verts, faces=new_faces, verts_colors=new_verts_colors)


def combine_meshes(
    meshes:List[Meshes],
) -> Meshes:
    """Combines multiple meshes into a single mesh.

    Args:
        meshes (List[Meshes]): List of meshes to combine.

    Returns:
        Meshes: Combined mesh.
    """
    all_verts = torch.zeros(0, 3, dtype=meshes[0].verts.dtype, device=meshes[0].verts.device)
    all_faces = torch.zeros(0, 3, dtype=meshes[0].faces.dtype, device=meshes[0].faces.device)
    all_verts_colors = None if meshes[0].verts_colors is None else torch.zeros(0, 3, dtype=meshes[0].verts_colors.dtype, device=meshes[0].verts_colors.device)
    
    n_total_verts = 0
    
    for _, mesh in enumerate(meshes):
        all_verts = torch.cat([all_verts, mesh.verts], dim=0)
        all_faces = torch.cat([all_faces, mesh.faces + n_total_verts], dim=0)
        if all_verts_colors is not None:
            all_verts_colors = torch.cat([all_verts_colors, mesh.verts_colors], dim=0)
        n_total_verts += mesh.verts.shape[0]
    
    return Meshes(verts=all_verts, faces=all_faces, verts_colors=all_verts_colors)


class RasterizationSettings():
    """
    Rasterization settings for meshes.
    """
    def __init__(
        self, 
        image_size=(1080, 1920),
        blur_radius=0.0,
        faces_per_pixel=1,
        ):
        self.image_size = image_size
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel


class Fragments():
    def __init__(self, bary_coords, zbuf, pix_to_face):
        self.bary_coords = bary_coords  # Shape (1, height, width, 1, 3)
        self.zbuf = zbuf  # Shape (1, height, width, 1)
        self.pix_to_face = pix_to_face  # Shape (1, height, width, 1)


class MeshRasterizer(torch.nn.Module):
    """
    Class for rasterizing meshes with NVDiffRast.
    """
    def __init__(
        self, 
        cameras:Union[List[Camera], Camera]=None,
        raster_settings:RasterizationSettings=None,
        use_opengl=True,
    ):
        super().__init__()
        
        if cameras is None:
            if raster_settings is None:
                raster_settings = RasterizationSettings()
            self.raster_settings = raster_settings
            self.height, self.width = raster_settings.image_size
            self.cameras = None
        else:
            if isinstance(cameras, Camera):
                cameras = [cameras]
            # Get height and width if provided in cameras
            self.height = cameras[0].image_height
            self.width = cameras[0].image_width
            self.raster_settings = RasterizationSettings(
                image_size=(self.height, self.width),
            )
            self.cameras = cameras
        
        if use_opengl:
            backend = os.environ.get("NVDIFRAST_BACKEND", "gl").lower()
            if backend == "cuda":
                self.gl_context = dr.RasterizeCudaContext()
            else:
                self.gl_context = dr.RasterizeGLContext()
        else:
            self.gl_context = dr.RasterizeCudaContext()
            
    def forward(
        self, 
        mesh:Meshes, 
        cameras:List[Camera]=None,
        cam_idx:int=0,
        return_only_pix_to_face:bool=False,
        return_rast_out:bool=False,
        return_positions:bool=False,
    ):
        if cameras is None:
            if self.cameras is None:
                raise ValueError("cameras must be provided either in the constructor or in the forward method")
            cameras = self.cameras
        
        if isinstance(cameras, Camera):
            render_camera = cameras
        else:
            render_camera = cameras[cam_idx]

        height, width = render_camera.image_height, render_camera.image_width
        nvdiff_rast_out = nvdiff_rasterization(
            camera=render_camera,
            image_height=height, 
            image_width=width,
            verts=mesh.verts,
            faces=mesh.faces,
            return_indices_only=False,
            glctx=self.gl_context,
            return_rast_out=return_rast_out,
            return_positions=return_positions,
        )
        bary_coords, zbuf, pix_to_face = nvdiff_rast_out[:3]
        if return_rast_out:
            rast_out = nvdiff_rast_out[3]
        if return_positions:
            pos = nvdiff_rast_out[4]
        
        if return_only_pix_to_face:
            return pix_to_face.view(1, height, width, 1)
        bary_coords = torch.cat([bary_coords, 1. - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)
        
        # TODO: Zbuf is still in NDC space, should convert to camera space
        fragments = Fragments(
            bary_coords.view(1, height, width, 1, 3),
            zbuf.view(1, height, width, 1),
            pix_to_face.view(1, height, width, 1),
        )
        _output = (fragments,)
        if return_rast_out:
            _output = _output + (rast_out,)
        if return_positions:
            _output = _output + (pos,)
        return _output
        
    
class MeshRenderer(torch.nn.Module):
    """
    Class for rendering meshes with NVDiffRast and a shader.
    """
    def __init__(self, rasterizer:MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        # TODO: Add shader
        
    def forward(
        self, 
        mesh:Meshes, 
        cameras:Union[List[Camera], Camera]=None, 
        cam_idx=0,
        return_depth=False,
        return_normals=False,
        use_antialiasing=True,
        return_pix_to_face=False,
        check_errors=True,
    ):
        fragments, rast_out, pos = self.rasterizer(mesh, cameras, cam_idx, return_rast_out=True, return_positions=True)
        if cameras is None:
            cameras = self.rasterizer.cameras
        if isinstance(cameras, Camera):
            cameras = [cameras]
        
        return_colors = mesh.verts_colors is not None

        output_pkg = {}
        
        # Compute per-vertex features to render
        features = torch.zeros(mesh.verts.shape[0], 0, device=mesh.verts.device)        
        
        if return_depth:
            depth_idx = features.shape[-1]
            verts_depth = transform_points_world_to_view(mesh.verts, [cameras[cam_idx]])[..., 2].squeeze()  # Shape (N, )
            features = torch.cat([features, verts_depth.view(mesh.verts.shape[0], 1)], dim=-1)
            
        if return_colors:
            color_idx = features.shape[-1]
            features = torch.cat([features, mesh.verts_colors], dim=-1)  # Shape (N, n_features)
        
        # Compute image
        feature_img, _ = dr.interpolate(features[None], rast_out, mesh.faces)  # Shape (1, H, W, n_features)
        
        # Antialiasing for propagating gradients
        if use_antialiasing:
            feature_img = dr.antialias(feature_img, rast_out, pos, mesh.faces)  # Shape (1, H, W, n_features)
        
        if return_depth:
            output_pkg["depth"] = feature_img[..., depth_idx:depth_idx+1]  # Shape (1, H, W)
        if return_colors:
            output_pkg["rgb"] = feature_img[..., color_idx:color_idx+3]  # Shape (1, H, W, 3)
            
        # Compute per-face normals
        if return_normals:
            valid_mask = fragments.pix_to_face >= 0  # Shape (1, H, W, 1)
            if check_errors:
                error_mask = fragments.pix_to_face >= mesh.faces.shape[0]
                error_encountered = torch.sum(error_mask)
                if error_encountered > 0:
                    print(f"[WARNING] Rasterized {error_encountered} pixels with invalid triangle index.")
                    fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=mesh.faces.shape[0] - 1)
                    valid_mask = valid_mask & ~error_mask
            output_pkg["normals"] = mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask  # Shape (1, H, W, 3)
            # if use_antialiasing:
            #     output_pkg["normals"] = dr.antialias(output_pkg["normals"], rast_out, pos, mesh.faces)  # Shape (1, H, W, 3)
            
        if return_pix_to_face:
            output_pkg["pix_to_face"] = fragments.pix_to_face

        return output_pkg


def fuse_fragments(fragments1:Fragments, fragments2:Fragments):
    raster_mask1 = fragments1.pix_to_face > -1
    raster_mask2 = fragments2.pix_to_face > -1
    
    # raster_mask1 = fragments1.zbuf > 0.
    # raster_mask2 = fragments2.zbuf > 0.
    
    no_raster_mask = (~raster_mask1) & (~raster_mask2)
    
    zbuf1 = torch.where(raster_mask1, fragments1.zbuf, 1000.)
    zbuf2 = torch.where(raster_mask2, fragments2.zbuf, 1000.)
    
    all_zbufs = torch.cat([zbuf1[..., None], zbuf2[..., None]], dim=-1)  # Shape (1, H, W, 1, 2)
    zbuf, argzbuf = torch.min(all_zbufs, dim=-1)  # argzbuf is of shape (1, H, W, 1)
    zbuf[no_raster_mask] = 0.
    
    all_pix_to_face = torch.cat(
        [
            fragments1.pix_to_face[..., None], 
            fragments2.pix_to_face[..., None]
        ], 
        dim=-1
    )  # Shape (1, H, W, 1, 2)
    pix_to_face = torch.gather(
        all_pix_to_face, 
        dim=-1, 
        index=argzbuf[..., None]
    )[..., 0]  # Shape (1, H, W, 1)
    pix_to_face[no_raster_mask] = -1
    
    all_bary_coords = torch.cat([fragments1.bary_coords[..., None], fragments2.bary_coords[..., None]], dim=-1)
    bary_coords = torch.gather(
        all_bary_coords, 
        dim=-1, 
        index=argzbuf[..., None, None].expand(-1, -1, -1, -1, 3, -1)
    )[..., 0]  # Shape (1, H, W, 1, 3)

    return Fragments(
        bary_coords,
        zbuf,
        pix_to_face,
    )

    
class ScalableMeshRenderer(torch.nn.Module):
    """
    Class for rendering big meshes with NVDiffRast.
    """
    def __init__(self, rasterizer:MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        # TODO: Add shader
        
    def forward(
        self, 
        mesh:Meshes, 
        cameras:Union[List[Camera], Camera]=None, 
        cam_idx:int=0,
        return_depth:bool=False,
        return_normals:bool=False,
        use_antialiasing:bool=True,
        return_pix_to_face:bool=False,
        check_errors:bool=True,
        max_triangles_in_batch:int=2**24  # Corresponds to the nb of triangles above which Nvdiffrast breaks
    ):
        n_passes = (mesh.faces.shape[0] + max_triangles_in_batch - 1) // max_triangles_in_batch
        
        fragments = None
        idx_shift = 0
        for i_pass in range(n_passes):
            start_idx = i_pass * max_triangles_in_batch
            end_idx = min(start_idx + max_triangles_in_batch, mesh.faces.shape[0])
            
            # Compute submesh
            sub_faces = mesh.faces[start_idx:end_idx]
            sub_mesh = Meshes(verts=mesh.verts, faces=sub_faces)
            
            # Rasterize submesh
            _fragments, _, pos = self.rasterizer(sub_mesh, cameras, cam_idx, return_rast_out=True, return_positions=True)
            
            # Combine fragments
            if fragments is None:
                fragments = _fragments
            else:
                # _fragments.pix_to_face = _fragments.pix_to_face + idx_shift
                _fragments.pix_to_face = torch.where(
                    _fragments.pix_to_face > -1, 
                    _fragments.pix_to_face + idx_shift, 
                    _fragments.pix_to_face
                )
                fragments = fuse_fragments(fragments, _fragments)
                # fragments = _fragments
            
            # Update idx shift
            idx_shift = idx_shift + len(sub_faces)
        
        # Filter mesh and fragments to keep only rasterized faces. This will decrease the number of faces to at most H * W.
        # Reducing the number of faces is necessary to avoid errors when running dr.interpolate and dr.antialias
        if True:
            filtered_face_idx, filtered_pix_to_face = fragments.pix_to_face.unique(return_inverse=True)
            filtered_face_idx = filtered_face_idx[1:]
            filtered_faces = mesh.faces[filtered_face_idx]
            filtered_pix_to_face = filtered_pix_to_face - 1
            mesh = Meshes(verts=mesh.verts, faces=filtered_faces, verts_colors=mesh.verts_colors)
            fragments.pix_to_face = filtered_pix_to_face

        if mesh.faces.shape[0] == 0:
            H, W = fragments.zbuf.shape[1:3]
            output_pkg = {}
            if return_depth:
                output_pkg["depth"] = torch.zeros(1, H, W, 1, device=mesh.verts.device)
            if return_normals:
                output_pkg["normals"] = torch.zeros(1, H, W, 3, device=mesh.verts.device)
            if return_pix_to_face:
                output_pkg["pix_to_face"] = fragments.pix_to_face
            return output_pkg
        
        # Rebuild rast_out
        rast_out = torch.zeros(*fragments.zbuf.shape[:-1], 4, device=fragments.zbuf.device)
        rast_out[..., :2] = fragments.bary_coords[..., 0, :2]
        rast_out[..., 2:3] = fragments.zbuf
        rast_out[..., 3:4] = fragments.pix_to_face.float() + 1
        
        if cameras is None:
            cameras = self.rasterizer.cameras
        if isinstance(cameras, Camera):
            cameras = [cameras]
        
        return_colors = mesh.verts_colors is not None

        output_pkg = {}
        
        # Compute per-vertex features to render
        features = torch.zeros(mesh.verts.shape[0], 0, device=mesh.verts.device)
        
        if return_depth:
            depth_idx = features.shape[-1]
            verts_depth = transform_points_world_to_view(mesh.verts, [cameras[cam_idx]])[..., 2].squeeze()  # Shape (N, )
            features = torch.cat([features, verts_depth.view(mesh.verts.shape[0], 1)], dim=-1)
            
        if return_colors:
            color_idx = features.shape[-1]
            features = torch.cat([features, mesh.verts_colors], dim=-1)  # Shape (N, n_features)
        
        # Short-circuit: no visible triangles -> return zero maps
        if mesh.faces.shape[0] == 0:
            H, W = fragments.zbuf.shape[1:3]
            if return_depth:
                output_pkg["depth"] = torch.zeros(1, H, W, 1, device=mesh.verts.device)
            if return_colors:
                output_pkg["rgb"] = torch.zeros(1, H, W, 3, device=mesh.verts.device)
            if return_normals:
                output_pkg["normals"] = torch.zeros(1, H, W, 3, device=mesh.verts.device)
            if return_pix_to_face:
                output_pkg["pix_to_face"] = fragments.pix_to_face
            return output_pkg

        # Compute image
        feature_img, _ = dr.interpolate(features[None], rast_out, mesh.faces)  # Shape (1, H, W, n_features)
        
        # Antialiasing for propagating gradients
        if use_antialiasing:
            feature_img = dr.antialias(feature_img, rast_out, pos, mesh.faces)  # Shape (1, H, W, n_features)
        
        if return_depth:
            output_pkg["depth"] = feature_img[..., depth_idx:depth_idx+1]  # Shape (1, H, W)

        if return_colors:
            output_pkg["rgb"] = feature_img[..., color_idx:color_idx+3]  # Shape (1, H, W, 3)
            
        # Compute per-face normals
        if return_normals:
            valid_mask = fragments.pix_to_face >= 0  # Shape (1, H, W, 1)
            if check_errors:
                error_mask = fragments.pix_to_face >= mesh.faces.shape[0]
                error_encountered = torch.sum(error_mask)
                if error_encountered > 0:
                    print(f"[WARNING] Rasterized {error_encountered} pixels with invalid triangle index.")
                    fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=mesh.faces.shape[0] - 1)
                    valid_mask = valid_mask & ~error_mask
            output_pkg["normals"] = mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask  # Shape (1, H, W, 3)
            # if use_antialiasing:
            #     output_pkg["normals"] = dr.antialias(output_pkg["normals"], rast_out, pos, mesh.faces)  # Shape (1, H, W, 3)
            
        if return_pix_to_face:
            output_pkg["pix_to_face"] = fragments.pix_to_face
            
        #### TO REMOVE
        output_pkg["fragments"] = fragments
        output_pkg["rast_out"] = rast_out
        #### TO REMOVE

        return output_pkg
