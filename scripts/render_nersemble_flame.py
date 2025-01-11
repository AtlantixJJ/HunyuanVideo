import os
import sys
import math
import torch
import nvdiffrast.torch
import kaolin as kal
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm
from PIL import Image
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLCogVideoX
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from kaolin.render.mesh.nvdiffrast_context import default_nvdiffrast_context
sys.path.append('.')
from lib.flame import FLAMEAPI
from lib.utils import imread_tensor



def tan2fov(tan):
    return 2 * math.atan(tan)

def scale_fov(old_fov, r0, r1):
    return 2 * math.atan(math.tan(old_fov/2) * r0 / r1)


def mesh_rasterize_interpolate_nvdiffrast(
        mesh, camera, nvdiffrast_context):
    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    if hasattr(camera.intrinsics, 'project'):
        vertices_clip = camera.intrinsics.project(vertices_camera)
    else:
        ones = torch.ones_like(vertices_camera[..., :1])
        vertices_clip = torch.cat([camera.intrinsics.transform(vertices_camera), ones], dim=-1)

    faces_int = mesh.faces.int()
    rast = nvdiffrast.torch.rasterize(nvdiffrast_context, vertices_clip, faces_int,
                                      (camera.height, camera.width), grad_db=False)
    # Nvdiffrast rasterization contains u, v, z/w, triangle_id onf shape 1 x W x H x 4
    rast0 = torch.flip(rast[0], dims=(1,))  # why filp?
    face_idx = (rast0[..., -1].long() - 1).contiguous()

    im_normals = im_uvs = im_tangents = im_features = None

    normals = mesh.vertex_normals
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-6)
    im_normals = nvdiffrast.torch.interpolate(normals, rast0, faces_int)[0]

    # depth rendering
    face_features = []
    fv_cam = kal.ops.mesh.index_vertices_by_faces(
        vertices_camera, mesh.faces)
    face_features.append(fv_cam[..., 2:])
    if mesh.has_or_can_compute_attribute('face_features'):
        face_features.append(mesh.face_features)
    face_features = torch.cat(face_features, dim=-1)
    val, idx = kal.ops.mesh.unindex_vertices_by_faces(face_features)
    im_features = nvdiffrast.torch.interpolate(
        val, rast0, idx.int())[0]
    im_depths = im_features[..., 0]
    im_features = im_features[..., 1:]

    if mesh.has_or_can_compute_attribute('uvs') and mesh.has_or_can_compute_attribute('face_uvs_idx'):
        im_uvs = nvdiffrast.torch.interpolate(
            mesh.uvs, rast0, mesh.face_uvs_idx.int())[0] % 1.

    if mesh.has_or_can_compute_attribute('vertex_tangents'):
        im_tangents = nvdiffrast.torch.interpolate(
            mesh.vertex_tangents, rast0, faces_int)[0]

    return face_idx, im_normals, im_depths, im_tangents, im_uvs, im_features


data_dir = 'data/NeRSemble/data'
instance_