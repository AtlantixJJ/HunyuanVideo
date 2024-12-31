"""Camera and related transformation.
"""
import math
import torch
import numpy as np
from PIL import Image
from typing import NamedTuple, Union, Optional
from scipy.spatial.transform import Rotation
from kaolin.render.camera import Camera as KCamera
from einops import rearrange, einsum

from lib.ops import matrix_to_quaternion

######################################
##### Gaussian Splatting Cameras #####
######################################


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    fov_x: float
    fov_y: float
    image: np.array
    image_path: str
    depth_path: str
    mask_path: str
    width: int
    height: int


class GSCamera:
    """Camera class for Gaussian Splatting (compatibility haven't been merged)."""

    def __init__(self,
                 world_view_transform=None, # world to camera
                 full_proj_transform=None, # world to screen
                 image=None, mask=None, depth=None,
                 image_path=None, mask_path=None, depth_path=None,
                 image_width=512, image_height=512,
                 fov_x=0.2443, fov_y=0.2443,
                 znear=0.01, zfar=100,
                 kao_cam=None):
        self.image, self.mask, self.depth = image, mask, depth
        self.image_width, self.image_height = image_width, image_height
        self.image_path = image_path
        self.mask_path, self.depth_path = mask_path, depth_path
        self.fov_x, self.fov_y = fov_x, fov_y
        self.znear, self.zfar = znear, zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.kao_cam = kao_cam
        if world_view_transform is not None:
            view_inv = torch.inverse(world_view_transform)
            self.camera_center = view_inv[3][:3]

    def __repr__(self) -> str:
        viz_fn = lambda x : '[' + ', '.join([f'{t:.3f}' for t in x]) + ']' \
            if isinstance(x, torch.Tensor) and x.ndim == 1 else f'{x:.3f}'

        #viz_fn = lambda x : x
        return f'GSCamera(FoVx={viz_fn(self.fov_x)} FoVy={viz_fn(self.fov_y)} world2cam={self.world_view_transform.shape} proj={self.full_proj_transform.shape} device={self.world_view_transform.device} image={self.image_path} depth={self.depth_path} mask={self.mask_path}\n'

    def to(self, device):
        """Move to device."""
        if self.image is not None:
            self.image = self.image.to(device)
        self.world_view_transform = self.world_view_transform.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self

    def as_dict(self):
        """Return a dictionary of attributes."""
        attrs = ['image_path', 'mask_path', 'depth_path',
                 'image', 'mask', 'depth',
                 'image_height', 'image_width',
                 'fov_x', 'fov_y', 'camera_center',
                 'world_view_transform', 'full_proj_transform']
        return {f'_camdata_{k}': getattr(self, k) for k in attrs \
                if getattr(self, k) is not None}

    @property
    def quat(self):
        cam2world = torch.linalg.inv(self.world_view_transform.T)
        return matrix_to_quaternion(cam2world[:3, :3])

    @staticmethod
    def from_dict(dic):
        """Load the attributes from a dictionary. Support a batch of cameras."""
        idx = len('_camdata_')
        if not isinstance(dic['_camdata_fov_x'], float):
            n_cams = len(dic['_camdata_fov_x'])
            cams = []
            for i in range(n_cams):
                cam = GSCamera()
                for k, v in dic.items():
                    setattr(cam, k[idx:], v[i])
                cams.append(cam)
        else:
            cam = GSCamera()
            for k, v in dic.items():
                if isinstance(v, list) and isinstance(v[0], list):
                    v = v[0]
                setattr(cam, k[idx:], v[i])
            cams = cam
        return cams

    def load_image(self):
        """Deprecated. Load the image only when requested."""
        if self.image is not None:
            return self.image
        image = np.array(Image.open(self.image_path))
        image = torch.from_numpy(image) / 255.0
        self.image = image.permute(2, 0, 1)
        self.image_width = self.image.shape[2]
        self.image_height = self.image.shape[1]
        return self.image

    def load_depth(self):
        """Load the depth only when requested."""
        if self.depth_path is None:
            return None
        if self.depth is not None:
            return self.depth
        depth = np.array(Image.open(self.depth_path))
        self.depth = torch.from_numpy(depth).float()
        return self.depth

    def load_mask(self):
        """Load the mask only when requested."""
        if self.mask_path is None:
            return None
        if self.mask is not None:
            return self.mask
        mask = np.array(Image.open(self.mask_path))
        self.mask = torch.from_numpy(mask).float()
        return self.mask

    @staticmethod
    def from_compact(c: torch.Tensor, **kwargs):
        """
        Args:
            c: [N, 25]. Extrinsics + Intrinsics.
        """
        cam2world_colmap = c[:, :16].reshape(-1, 4, 4)
        world2cam_colmap = torch.linalg.inv(cam2world_colmap)
        intrinsics = c[:, 16:].reshape(-1, 3, 3)
        fov_xs = 2 * torch.atan(1 / intrinsics[:, 0, 0])
        fov_ys = 2 * torch.atan(1 / intrinsics[:, 1, 1])
        #print(world2cam_colmap @ torch.Tensor([0, 0, 0, 1]).to(world2cam_colmap))
        return [GSCamera.from_matrix(w2c, float(fov_x), float(fov_y), **kwargs)
                for w2c, fov_x, fov_y in \
                zip(world2cam_colmap, fov_xs, fov_ys)]

    def to_compact(self):
        """
        Args:
            c: [N, 25]. Extrinsics + Intrinsics.
        """
        cam2world_colmap = torch.linalg.inv(self.world_view_transform.T)
        intrinsics = intrinsics_from_fov(self.fov_x, self.fov_y).to(cam2world_colmap)
        return torch.cat([cam2world_colmap.reshape(-1), intrinsics.view(-1)])

    @staticmethod
    def to_extrinsics_intrinsics(cameras, device=None):
        """
        Returns:
            {
                image: [N, 1, C, H, W], batch view channel H W
                extrinsics: [N, 1, 4, 4],
                intrinsics: [N, 1, 3, 3],
                near: [N, 1],
                far: [N, 1],
            }
        """
        dic = {}
        device = device if device is not None else \
            cameras[0].world_view_transform.device
        #ones = torch.ones((len(cameras), 1)).to(device)
        intrinsics = lambda x: torch.Tensor([
            [0.5 / math.tan(x/2), 0, 0.5],
            [0, 0.5 / math.tan(x / 2), 0.5],
            [0, 0, 1]])
        dic['image'] = None if cameras[0].image is None else \
            torch.stack([c.image for c in cameras]).to(device)
        E = torch.stack([
            torch.linalg.inv(c.world_view_transform.T)
            for c in cameras]).to(device)
        E[..., :3, :3] /= torch.det(E[..., :3, :3])[..., None, None] ** (1/3)
        dic['extrinsics'] = E
        dic['intrinsics'] = torch.stack([
            intrinsics(c.fov_x) for c in cameras]).to(device)
        return dic

    @staticmethod
    def from_extrinsics_intrinsics(extrinsics, intrinsics, **kwargs):
        fov_x, fov_y = intrinsics_to_fov(intrinsics)
        world_view_transform = torch.linalg.inv(extrinsics)
        return GSCamera.from_matrix(world_view_transform, fov_x, fov_y, **kwargs)

    @staticmethod
    def from_info(cam_info: CameraInfo):
        """Creates a Camera object from a CameraInfo."""
        world2view = world2view_from_rt(cam_info.R, cam_info.T)
        world_view_transform = torch.Tensor(world2view).transpose(0, 1)
        K = intrinsics_from_fov(cam_info.fov_x, cam_info.fov_y)
        K_full = full_perspective_matrix(K)
        full_proj_transform = world_view_transform[None].bmm(K_full.T[None])[0]
        return GSCamera(world_view_transform, full_proj_transform,
                        fov_x=cam_info.fov_x, fov_y=cam_info.fov_y,
                        image_path=cam_info.image_path,
                        depth_path=cam_info.depth_path,
                        mask_path=cam_info.mask_path)

    @staticmethod
    def from_matrix(world_view_transform, fov_x, fov_y, **kwargs):
        """Creates a Camera object from a CameraInfo.
        Args:
            FoV: Field of view in radian.
        """

        #proj_matrix = perspective_matrix_colmap(
        #    znear=0.01, zfar=100,
        #    fov_x=fov_x, fov_y=fov_y).T.to(world_view_transform)
        K = intrinsics_from_fov(fov_x, fov_y)
        proj_matrix = full_perspective_matrix(K).T.to(world_view_transform)
        full_proj_transform = world_view_transform.T @ proj_matrix
        return GSCamera(world_view_transform.T, full_proj_transform,
                        fov_x=fov_x, fov_y=fov_y, **kwargs)


def to_homogenized(points):
    """Convert points to homogenized coordinates."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def from_homogenized(points):
    """Convert points from homogenized coordinates."""
    return points[..., :-1] / points[..., -1:]


def world2cam_projection(w2c, intrinsics, points):
    """Convert world to camera space and project to screen space.
    Args:
        w2c: torch.Tensor, [B, 4, 4], world to camera matrix. (Already transposed!)
        intrinsics: torch.Tensor, [B, 3, 3], intrinsics matrix.
        points: torch.Tensor, [B, N, 3], world space points.
    Returns:
        torch.Tensor, [B, N, 2], screen space points.
    """
    cam_points = from_homogenized(to_homogenized(points) @ w2c)
    cam_points = cam_points @ intrinsics.permute(0, 2, 1)
    return from_homogenized(cam_points)


def unproject(
    coordinates, # (..., 3), :2 are x, y, 2 is z
    intrinsics, # (..., 3, 3)
    extrinsics, # (..., 4, 4)
):
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    z = coordinates[..., 2:].clone() # (L,)
    coordinates[..., 2] = 1
    ray_directions = einsum(
        intrinsics.inverse(), coordinates,
        "... i j, ... j -> ... i") # (..., 3)
    coordinates[..., 2:] = z
    # Apply the supplied depth values.
    points = ray_directions * z

    # Apply extrinsics transformation
    return einsum(extrinsics[..., :3, :3], points, '... i j, ... j -> ... i') \
            + extrinsics[..., :3, 3]


def world2view_from_rt(R, t):
    """Get world to view matrix from rotation and translation.
    Args:
        R: torch.Tensor, [3, 3], rotation matrix.
        t: torch.Tensor, [3, ], translation vector.
    """
    if isinstance(R, np.ndarray):
        Rt = np.zeros((4, 4))
    else:
        Rt = torch.zeros((4, 4))

    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    if isinstance(R, np.ndarray):
        return Rt.astype('float32')
    return Rt.float()


def perspective_matrix_colmap(znear, zfar, fov_x, fov_y):
    """In colmap coordinate system, convert a frustum to NDC space in [-1, 1].
    """
    tanHalfFovY = math.tan(fov_y / 2)
    tanHalfFovX = math.tan(fov_x / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    # normalize world space x to NDC x, scale x by (znear / z)
    P[0, 0] = 2.0 * znear / (right - left)
    # shift x to the center
    P[0, 2] = (right + left) / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def full_perspective_matrix(K, znear=0.01, zfar=100):
    """In colmap coordinate system, convert a frustum to NDC space in [-1, 1].
    """

    fx = K[0, 0] * 2
    fy = K[1, 1] * 2
    x0 = K[0, 2] - 0.5
    y0 = K[1, 2] - 0.5

    tx = ty = 0 # the center of NDC space, 0
    lx = ly = 2 # the boundary length, -1 to 1 is 2
    z_sign = 1.0
    U = z_sign * zfar / (zfar - znear)
    V = -(zfar * znear) / (zfar - znear)

    return torch.Tensor([
        [2 * fx / lx, 0, -2 * x0 / lx + tx, 0],
        [0, 2 * fy / ly, -2 * y0 / ly + ty, 0],
        [0, 0, U, V],
        [0, 0, z_sign, 0]
    ])


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def intrinsics_to_fov(intrinsics):
    """Convert intrinsics to FoV.
    Args:
        intrinsics: a tensor of shape [batch, 3, 3]
    Returns:
        FoVx, FoVy: a tensor of shape [batch], in radians.
    """
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    return 2 * torch.atan2(cx, fx), 2 * torch.atan2(cy, fy)


def intrinsics_from_fov(
        fov_x: float,
        fov_y: float,
        size=1.0):
    """Get the camera intrinsics matrix from FoV.
    Notice that this transforms points into screen space [0, size]^2 rather than NDC.
    .----> x
    |
    |
    v y
    Args:
        fov_x: Field of View in x-axis (radians).
        fov_y: Field of View in y-axis (degrees).
        znear: near plane.
        zfar: far plane.
    """
    fx = 0.5 * size / math.tan(fov_y / 2)
    fy = 0.5 * size / math.tan(fov_x / 2)

    return torch.Tensor([
        [fx, 0, 0.5 * size],
        [0, fy, 0.5 * size],
        [0, 0, 1]])


def kaolin2colmap_cam2world(kaolin_cam2world: torch.Tensor):
    """Convert a Kaolin camera to world matrix to a Colmap matrix.
    COLMAP (OpenCV)'s cooredinate system uses -y and -z than Kaolin.
    """
    sign = torch.Tensor([1, -1, -1, 1]).expand_as(kaolin_cam2world)
    return kaolin_cam2world * sign.to(kaolin_cam2world)


def kaolin2colmap_cam(kaolin_cam: KCamera):
    """Convert a Kaolin camera to a Colmap camera."""
    fov_x = float(kaolin_cam.intrinsics.fov_x) / 180 * math.pi # degree
    fov_y = float(kaolin_cam.intrinsics.fov_y) / 180 * math.pi
    #proj_matrix = perspective_matrix_colmap(
    #    znear=1e-2, zfar=1e2, fov_x=fov_x, fov_y=fov_y).transpose(0, 1)
    K = intrinsics_from_fov(fov_x, fov_y)
    proj_matrix = full_perspective_matrix(K).transpose(0, 1)

    kaolin_cam2world = kaolin_cam.extrinsics.inv_view_matrix()[0]
    colmap_cam2world = kaolin2colmap_cam2world(kaolin_cam2world)
    colmap_world2cam = colmap_cam2world.inverse().transpose(0, 1)
    colmap_fullproj = colmap_world2cam @ proj_matrix.to(colmap_world2cam)
    return GSCamera(colmap_world2cam, colmap_fullproj,
                    fov_x=fov_x, fov_y=fov_y,
                    image_width=kaolin_cam.width,
                    image_height=kaolin_cam.height,
                    kao_cam=kaolin_cam)


def unproject_depth(depth, cam2world_matrix, intrinsics):
    """
    Convert a depth image DEPTH to points in world space.

    depth: (N, resolution, resolution)
    cam2world_matrix: (N, 4, 4)
    intrinsics: (N, 3, 3)
    """
    resolution = depth.shape[-1]
    N, M = depth.shape[0], resolution ** 2
    # ij indexing: first column expansion, second row expansion
    uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), 
            torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), 
            indexing='ij')) * (1. / resolution) + (0.5 / resolution)
    uv = uv.reshape(2, -1).transpose(1, 0)
    uv = uv.unsqueeze(0).repeat(N, 1, 1) # shape: (N, M, 2)

    xyz = torch.linalg.solve(intrinsics, torch.concat((uv, 
            torch.ones(N, M, 1).to(intrinsics.device)), dim=-1).permute(0, 2, 1)) 

    cam_rel_points = torch.concat((xyz * depth.view(N, 1, M), torch.ones(N, 1, M).to(intrinsics.device)), dim=1) # shape: (N, 4, M)

    world_points = torch.bmm(cam2world_matrix, cam_rel_points).permute(0, 2, 1)[:, :, :3]

    return world_points


def angle2matrix(angles):
    cos_y, cos_x, cos_z = torch.cos(angles).T
    sin_y, sin_x, sin_z = torch.sin(angles).T
    R = torch.eye(3)[None].repeat(angles.shape[0], 1, 1).to(angles)

    Rx = R.clone() # rotate around x: pitch
    Rx[:, 1, 1] = cos_x
    Rx[:, 1, 2] = -sin_x
    Rx[:, 2, 1] = sin_x
    Rx[:, 2, 2] = cos_x

    Ry = R.clone() # rotate around y: yaw
    Ry[:, 0, 0] = cos_y
    Ry[:, 0, 2] = sin_y
    Ry[:, 2, 0] = -sin_y
    Ry[:, 2, 2] = cos_y

    Rz = R.clone() # rotate around z: roll
    Rz[:, 0, 0] = cos_z
    Rz[:, 0, 1] = -sin_z
    Rz[:, 1, 0] = sin_z
    Rz[:, 1, 1] = cos_z
    # yaw -> pitch -> roll
    return torch.bmm(Rz, torch.bmm(Rx, Ry))


def matrix2angle(R):
    """Convert rotation matrix to angles."""
    # [yaw, pitch, roll]
    return Rotation.from_matrix(R.cpu().numpy()).as_euler('yxz')


def make_colmap_camera(angles, radius, fov):
    """
    Args:
        angles: (N, 3), [yaw, pitch, roll]
    """
    R = angle2matrix(angles)
    # build world2cam matrix
    world2cams = torch.eye(4)[None].repeat(angles.shape[0], 1, 1)
    world2cams[:, :3, :3] = R
    world2cams[:, 2, 3] = -radius
    cam2worlds = torch.linalg.inv(world2cams)
    cam2worlds[..., 1] *= -1
    cam2worlds[..., 2] *= -1
    world2cams = torch.linalg.inv(cam2worlds)
    return [GSCamera.from_matrix(w2c, fov, fov) for w2c in world2cams]


def pos_from_angle(
        azim: torch.Tensor,
        elev: torch.Tensor,
        radius: torch.Tensor):
    """Create point from angles and radius."""
    cos_elev = torch.cos(elev)
    x = cos_elev * torch.sin(azim)
    z = cos_elev * torch.cos(azim)
    y = torch.sin(elev)
    return radius * torch.stack([x, y, z], -1)


def sample_delta_angle(
        azim_std: float = 0.,
        elev_std: float = 0.,
        roll_std: float = 0.,
        n_sample: int = 1,
        device: str = 'cpu'):
    """
    Sample a delta angle from a Gaussian or uniform distribution.
    Args:
        azim: azimuthal angle (rotation around y axis) in radians
        elev: polar angle (angle from the y axis) in radians
        azim_std: standard deviation of azimuthal angle in radians
        elev_std: standard deviation of polar angle in radians
        n_sample: number of samples to return
        device: device to put output on
        noise: 'gaussian' or 'uniform'
    """
    dh = torch.rand((n_sample,), device=device) * azim_std if azim_std > 0 \
        else torch.zeros((n_sample,), device=device)
    dv = torch.randn((n_sample,), device=device) * elev_std if elev_std > 0 \
        else torch.zeros((n_sample,), device=device)
    dr = torch.randn((n_sample,), device=device) * roll_std if roll_std > 0 \
        else torch.zeros((n_sample,), device=device)
    return dh, dv, dr


def sample_lookat_camera(
        look_at: torch.Tensor = None,
        azim: float = 0, elev: float = 0,
        azim_std: float = 0., elev_std: float = 0.,
        cam_poses: torch.Tensor = None,
        fov: float = 0.2443,
        radius: float = 2.7,
        n_sample: int = 1,
        resolution: int = 512,
        device: str = 'cpu'):
    """
    Sample a camera pose looking at a point with a Gaussian or uniform distribution.
    Args:
        azim: azimuthal angle (rotation around y axis) in radians
        elev: polar angle (angle from the y axis) in radians
        look_at: 3-vector, point to look at
        azim_std: standard deviation of azimuthal angle in radians
        elev_std: standard deviation of polar angle in radians
        fov: field of view in radians
        radius: distance from camera to look_at point
        n_sample: number of samples to return
        resolution: image resolution of the camera
        device: device to put output on
        noise: 'gaussian' or 'uniform'
    """

    if cam_poses is None:
        dh, dv = sample_delta_angle(azim_std, elev_std, 0, n_sample, device)[:2]
        h = dh + azim
        v = dv + elev
        cam_poses = pos_from_angle(h, v, radius)
    if look_at is None:
        look_at = torch.Tensor([0., 0., 0.]).to(device)
    common_kwargs = {
        'at': look_at.to(device),
        'up': torch.Tensor([0., 1., 0.]).to(device),
        'fov': fov,
        'width': resolution,
        'height': resolution,
        'device': device
    }

    cams = [kaolin2colmap_cam(KCamera.from_args(
            eye=x, **common_kwargs)) for x in cam_poses]
    return cams
