"""Gaussian multiview images dataset."""
import os
import torch
import trimesh
import numpy as np
from plyfile import PlyData

from lib.gaussian_splatting import GaussianParameters, RGB2SH
from lib.dataset.multiview import MultiViewDataset, osj, cat_dict


def get_dict_normalization(g_dic):
    """
    Args:
        g_dic: TensorDict.
    """
    norm_dic = {}
    for k, v in g_dic.tensor_dic().items():
        norm_dic[k] = {
            'mean': v.mean(dim=0),
            'std': v.std(dim=0).clamp(min=1e-6),
        }
    concat_mean = torch.cat([v['mean'].view(-1) for v in norm_dic.values()])
    concat_std = torch.cat([v['std'].view(-1) for v in norm_dic.values()])
    return norm_dic, {'mean': concat_mean, 'std': concat_std}


def normalize_dict(g_dic, norm_dic):
    """
    Args:
        g_dic: dict of tensors.
        norm_dic: dict of normalization parameters.
    """
    for k, v in g_dic.tensor_dic().items():
        dims = [1] * (v.ndim - 1)
        mu = norm_dic[k]['mean'].view(*dims, -1)
        sigma = norm_dic[k]['std'].view(*dims, -1)
        g_dic[k] = (v - mu) / sigma
    return g_dic


def normalize_vector(v, norm_dic):
    """
    Args:
        v: tensor.
        norm_dic: dict of normalization parameters.
    """
    dims = [1] * (v.ndim - 1)
    mu = norm_dic['mean'].view(*dims, -1)
    sigma = norm_dic['std'].view(*dims, -1)
    return (v - mu) / sigma


def denormalize_vector(v, norm_dic):
    """
    Args:
        v: tensor.
        norm_dic: dict of normalization parameters.
    """
    dims = [1] * (v.ndim - 1)
    mu = norm_dic['mean'].view(*dims, -1)
    sigma = norm_dic['std'].view(*dims, -1)
    return v * sigma + mu


def denormalize_dict(g_dic, norm_dic):
    """
    Args:
        g_dic: dict of tensors.
        norm_dic: dict of normalization parameters.
    """
    for k, dic in norm_dic.items():
        dims = [1] * (g_dic[k].ndim - 1)
        mu = dic['mean'].view(*dims, -1)
        sigma = dic['std'].view(*dims, -1)
        g_dic[k] = g_dic[k] * sigma + mu
    return g_dic




def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    #print("Number of points at loading : ", xyz.shape[0])

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    max_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5 - 1)
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, opacities, scales, rots, max_sh_degree, features_extra


def load_init_ply(path):
    pcd = trimesh.load(path)
    points = pcd.vertices
    colors = pcd.colors[:, :3]
    n_points = points.shape[0]

    points = torch.tensor(points).float()
    # not sure why but the initialization is an inverse color
    # (N, 1, 3)
    sh = RGB2SH(torch.tensor(colors).float() / 255.)[..., None, :] 

    scale_unit = 6 - torch.log(torch.tensor(n_points)) # empirical selection
    scales = scale_unit * torch.ones((n_points, 3), dtype=torch.float32)
    rots = torch.zeros((n_points, 4), dtype=torch.float32)
    rots[:, 0] = 1
    opacities = torch.zeros((n_points, 1), dtype=torch.float32)
    return points, sh, opacities, scales, rots, 0, None


def convert_gaussian(xyz, features_dc, opacities, scales, rots):
    return GaussianParameters(
        active_sh_degree=0, max_sh_degree=0,
        xyz=xyz.clone().detach(),
        sh=features_dc.clone().detach(),
        opacity=torch.sigmoid(opacities).clone().detach(),
        scale=torch.exp(scales).clone().detach(),
        rotation=rots.clone().detach())


class GaussianMVDataset(MultiViewDataset):
    def __init__(self, root_dir: str, ann_file: str, use_depth=False,
        max_size=-1, sample_acam=2.7, n_points=-1,
        n_ref=1, n_tarfar=16, n_tarnear=16,
        resolution=-1, mode_fit=False):
        super().__init__(
            root_dir, ann_file, use_depth, max_size, sample_acam, n_ref, n_tarfar, n_tarnear, resolution)
        self.n_points = n_points
        self.mode_fit = mode_fit
        if not mode_fit:
            self.check_data_list()

    def __len__(self):
        if self.mode_fit:
            return len(self.scene_df)
        return len(self.valid_scene_indices)

    def check_data_list(self):
        """Check whether gaussians.pth is present in each scene."""
        self.valid_scene_indices = []
        for idx in range(len(self.scene_df)):
            scene_name = self.get_scene_name(idx)
            fpath = osj(self.root_dir, scene_name, 'gaussians.pth')
            if os.path.exists(fpath):
                self.valid_scene_indices.append(idx)

    def get_scene_name(self, idx):
        return self.scene_df.iloc[int(idx)]['capture']

    def __getitem__(self, idx):
        if not self.mode_fit:
            idx = self.valid_scene_indices[idx]
        if self.max_size > 0:
            idx = idx % self.max_size

        scene_df = self.scene_df.iloc[idx]
        scene_name = scene_df['capture']
        scene_prefix = osj(self.root_dir, scene_name)
        if not os.path.exists(scene_prefix):
            scene_prefix = self.root_dir

        gaussian_path = osj(scene_prefix, 'gaussians.pth')
        scene_image_fpaths = scene_df['image_path']
        n_cam = len(scene_image_fpaths)

        world2cam = scene_df['world2cam'].reshape(-1, 4, 4)
        cam2world = np.linalg.inv(world2cam)
        cam_pos = cam2world[:, :3, 3]
        ref_cam_indices, tarnear_cam_indices, tarfar_cam_indices = \
            self.choose_cameras(cam_pos)

        if self.max_size == 1:
            if not hasattr(self, 'cache_cameras'):
                self.cache_cameras = [
                    self.get_camera_dict(scene_prefix, scene_df, cam_idx)
                    for cam_idx in np.arange(n_cam)]
                sd = torch.load(gaussian_path, map_location='cpu')
                self.cache_gaussian = GaussianParameters(sd).requires_grad_(False)
            get_camera_fn = lambda cam_idx: self.cache_cameras[cam_idx]
            gaussians = self.cache_gaussian
        else:
            get_camera_fn = lambda cam_idx: self.get_camera_dict(
                scene_prefix, scene_df, cam_idx)
            if os.path.exists(gaussian_path):
                sd = torch.load(gaussian_path, map_location='cpu')
                gaussians = GaussianParameters(sd).requires_grad_(False)
            else: # temporary fix
                res = load_init_ply(osj(scene_prefix, 'point_cloud_init.ply'))[:-2]
                gaussians = convert_gaussian(*res)

        if self.n_points > 0:
            n_points = gaussians.xyz.shape[0]
            if n_points > self.n_points:
                indices = torch.randperm(n_points)[:self.n_points]
            elif n_points == self.n_points:
                indices = torch.arange(n_points)
            else:
                indices = torch.randint(n_points, (self.n_points,))
            gaussians = gaussians.slice_indice(indices)

        ref_cameras = [get_camera_fn(i) for i in ref_cam_indices]
        tarnear_cameras = [get_camera_fn(i) for i in tarnear_cam_indices]
        tarfar_cameras = [get_camera_fn(i) for i in tarfar_cam_indices]

        # (N_VIEWS, 3, H, W)
        dic = {
            'ref_camera': cat_dict([cam.as_dict() for cam in ref_cameras]),
            '3dgs_param': gaussians.valid_dic(),
            '__scene_idx': self.scene2idx[scene_name],
        }

        if self.n_tarfar:
            dic['tarfar_camera'] = cat_dict([cam.as_dict() for cam in tarfar_cameras])
        if self.n_tarnear:
            dic['tarnear_camera'] = cat_dict([cam.as_dict() for cam in tarnear_cameras])
        return dic


if __name__ == "__main__":
    # compute normalization coefficients
    from tqdm import tqdm
    torch.set_grad_enabled(False)
    data_dir = 'data/PanoHeadDense'
    ann_file = 'data/PanoHeadDense/ann.parquet'
    ds = GaussianMVDataset(data_dir, ann_file)

    norm_dics, normvec_dics = [], []
    for idx in tqdm(range(len(ds))):
        scene_df = ds.scene_df.iloc[idx]
        scene_name = scene_df['capture']
        scene_prefix = osj(ds.root_dir, scene_name)
        if not os.path.exists(scene_prefix):
            scene_prefix = ds.root_dir
        gaussian_path = osj(scene_prefix, 'gaussians.pth')
        if os.path.exists(gaussian_path):
            sd = torch.load(gaussian_path, map_location='cpu')
            gaussians = GaussianParameters(sd).requires_grad_(False)
        norm_dic, normvec_dic = get_dict_normalization(gaussians)
        norm_dics.append(norm_dic)
        normvec_dics.append(normvec_dic)
    # merge normalization coefficients
    norm_dic = {}
    for k in norm_dics[0].keys():
        mu = torch.stack([d[k]['mean'] for d in norm_dics]).mean(dim=0)
        var = torch.stack([d[k]['std'] ** 2 for d in norm_dics]).sum(dim=0)
        # add mean adjustment of std
        var += torch.stack([(d[k]['mean'] - mu) ** 2 for d in norm_dics]).sum(dim=0)
        std = (var / len(norm_dics)).sqrt()
        norm_dic[k] = {'mean': mu, 'std': std}
    torch.save(norm_dic, 'data/PanoHeadDense/normalization.pth')