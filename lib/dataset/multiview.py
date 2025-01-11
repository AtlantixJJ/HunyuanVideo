import os
import json
import glob
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, Lambda, CenterCrop
from os.path import join as osj

from lib.ops import farthest_point_sampling
from lib.utils import imread_pil
from lib.camera import GSCamera, CameraInfo, focal2fov, fov2focal, world2view_from_rt, focal2fov, kaolin2colmap_cam2world
from lib.utils import cat_dict, imread_pil


DF_COLUMNS = ['capture', 'world2cam', 'intrinsics', 'image_path', 'mask_path', 'depth_path']


def intrinsics_from_fov(
        fov_x: float,
        fov_y: float):
    """Get the camera intrinsics matrix from FoV.
    Args:
        fov: Field of View in radian.
    """
    fx = 1 / np.tan(fov_y / 2)
    fy = 1 / np.tan(fov_x / 2)

    return torch.Tensor([
        [fx, 0, 0.5],
        [0, -fy, 0.5],
        [0, 0, 1]
    ])


class MultiViewDataset(torch.utils.data.Dataset):
    """Multi-view dataset.
    """

    def __init__(self, root_dir: str, ann_file: str, use_depth=False,
                 max_size=-1, sample_acam=1, n_ref=1, n_tarfar=16, n_tarnear=16, resolution=512):
        """
        Args:
            root_dir: str, root directory of the dataset.
            ann_file: str, annotation file.
            max_size: int, maximum number of samples.
            sample_acam: float, sample around camera, quaternion threshold.
            resolution: int, resolution of the images.
        """
        self.ann_file = ann_file
        self.root_dir = root_dir
        self.resolution = resolution
        self.n_ref = n_ref
        self.n_tarfar = n_tarfar
        self.n_tarnear = n_tarnear
        self.use_depth = use_depth
        self.sample_acam = sample_acam # sample around camera, L2 threshold
        self.scene_df = pd.read_parquet(ann_file)
        self.max_size = max_size
        capture_names = self.scene_df['capture'].unique()
        if self.max_size > 0:
            capture_names = capture_names[:self.max_size]
        self.scene2idx = {name: idx for idx, name in enumerate(capture_names)}
        #self.idx2scene = {idx: name for name, idx in self.scene2idx.items()}
        
        ops = [Lambda(lambda x: CenterCrop(min(x.size))(x))]
        if resolution > 0:
            ops.append(Resize((resolution, resolution)))
        ops.append(ToTensor())
        self.transforms = Compose(ops)

    def __len__(self):
        return len(self.scene_df)

    def get_camera_dict(self, scene_prefix, scene_dic, cam_idx):
        """Get camera dictionary."""
        world2cam = scene_dic['world2cam'].reshape(-1, 4, 4)[cam_idx]
        intrinsic = scene_dic['intrinsics'].reshape(-1, 3, 3)[cam_idx]
        film_half_width = max(intrinsic[:2, 2])
        fov = focal2fov(intrinsic[0, 0], film_half_width * 2)
        image_path = osj(scene_prefix, scene_dic['image_path'][cam_idx])
        mask_path = osj(scene_prefix, scene_dic['mask_path'][cam_idx])
        image = self.transforms(imread_pil(image_path))
        mask = self.transforms(imread_pil(mask_path))[:1]
        width, height = image.shape[1:]

        if self.use_depth:
            depth_path = osj(scene_prefix, scene_dic['depth_path'][cam_idx])
            depth = np.asarray(Image.open(open(depth_path, "rb")))
            # need to use scale invariant loss
            depth_min, depth_max = 2.1, 3.3 # this number is verified
            depth = depth / 65535.0 * (depth_max - depth_min) + depth_min
            depth = self.transforms(Image.fromarray(depth))
        else:
            depth = torch.zeros_like(mask)

        return GSCamera.from_matrix(
                torch.from_numpy(world2cam).float(),
                fov_x=fov, fov_y=fov,
                image=image, mask=mask, depth=depth,
                image_width=width, image_height=height,
                image_path=image_path)

    def choose_cameras(self, cam_pos):
        n_cam = cam_pos.shape[0]
        if self.n_ref == -1:
            ref_cam_indices = list(range(n_cam))
        else:
            ref_cam_indices = farthest_point_sampling(cam_pos, self.n_ref)

        # find near target cameras
        diff = cam_pos[:, None] - cam_pos[ref_cam_indices][None]
        dist = np.sqrt((diff ** 2).sum(-1))
        dist[dist < 1e-3] = 1e6 # remove self
        if self.n_tarnear == -1:
            tarnear_cam_indices = list(range(n_cam))
        elif self.n_tarnear > 0:
            tarnear_cam_indices = [0] * self.n_tarnear
            for i in range(self.n_tarnear):
                ref_idx = i % self.n_ref
                #near_inds = dist[ref_idx] < dist[ref_idx].min() * (1 + self.sample_acam)
                near_inds = dist[:, ref_idx] < self.sample_acam

                near_inds = near_inds.nonzero()[0]
                if near_inds.shape[0] == 0:
                    near_inds = list(range(dist.reshape(-1).shape[0]))
                tarnear_cam_indices[i] = int(np.random.choice(near_inds))
        else:
            tarnear_cam_indices = []

        # other target cameras are used for union gaussian inference
        if self.n_tarfar == -1:
            tarfar_cam_indices = list(range(n_cam))
        elif self.n_tarfar > 0:
            tarfar_cam_indices = np.random.choice(
                np.arange(n_cam), self.n_tarfar,
                replace=n_cam < self.n_tarfar)
        else:
            tarfar_cam_indices = []
        return ref_cam_indices, tarnear_cam_indices, tarfar_cam_indices

    def __getitem__(self, idx):
        """
        Returns:
            tar_camera: n_views cameras. Each corresponding to the closest ref camera.
            ref_camera: n_views cameras. 
        """
        # notice: we randomize here to avoid resetting every epoch
        #idx = np.random.randint(len(self.sample_indice))
        #scene_idx, tar_cam_idx = self.sample_indice[idx]

        if self.max_size > 0:
            idx = idx % self.max_size

        scene_df = self.scene_df.iloc[int(idx)]
        scene_name = scene_df['capture']
        scene_prefix = osj(self.root_dir, scene_name)
        if not os.path.exists(scene_prefix):
            scene_prefix = self.root_dir

        world2cam = scene_df['world2cam'].reshape(-1, 4, 4)
        cam2world = np.linalg.inv(world2cam)
        cam_pos = cam2world[:, :3, 3]

        ref_cam_indices, tarnear_cam_indices, tarfar_cam_indices = \
            self.choose_cameras(cam_pos)

        ref_cameras = [self.get_camera_dict(scene_prefix, scene_df, cam_idx)
                       for cam_idx in ref_cam_indices]
        tarnear_cameras = [self.get_camera_dict(scene_prefix, scene_df, cam_idx)
                       for cam_idx in tarnear_cam_indices]
        tarfar_cameras = [self.get_camera_dict(scene_prefix, scene_df, cam_idx)
                       for cam_idx in tarfar_cam_indices]

        # all the tensors get stacked after dataloader
        # for convenience we will stack them first
        # (N_VIEWS, 3, H, W)
        dic = {
            'ref_camera': cat_dict([cam.as_dict() for cam in ref_cameras]),
            '__scene_idx': self.scene2idx[scene_name],
        }
        if self.n_tarfar: # can be -1, which is all cameras
            dic['tarfar_camera'] = cat_dict([cam.as_dict() for cam in tarfar_cameras])
        if self.n_tarnear:
            dic['tarnear_camera'] = cat_dict([cam.as_dict() for cam in tarnear_cameras])
        return dic
