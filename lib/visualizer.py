# pylint: skip-file
"""All kinds of visualizations."""
import math
import torch
import trimesh
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image
from typing import List, Union
from moviepy.editor import ImageSequenceClip
from einops import rearrange

from lib.gaussian_splatting import splat_gaussians, GaussianParameters
from lib.camera import sample_lookat_camera
from lib.basetype import TensorDict


#################################
######## Creating Images ########
#################################


POSITIVE_COLOR = matplotlib.colormaps["Reds"]
NEGATIVE_COLOR = matplotlib.colormaps["Blues"]
def heatmap_numpy(image):
    """Get the heatmap of the image

    Args:
        image : A numpy array of shape (N, H, W) and scale in [-1, 1]

    Returns:
        A image of shape (N, H, W, 3) in [0, 1] scale
    """
    image1 = image.copy()
    mask1 = image1 > 0
    image1[~mask1] = 0

    image2 = -image.copy()
    mask2 = image2 > 0
    image2[~mask2] = 0

    pos_img = POSITIVE_COLOR(image1)[:, :, :, :3]
    neg_img = NEGATIVE_COLOR(image2)[:, :, :, :3]

    x = np.ones_like(pos_img)
    x[mask1] = pos_img[mask1]
    x[mask2] = neg_img[mask2]

    return x


def heatmap_torch(image):
    """Torch tensor version of heatmap_numpy.

    Args:
        image : torch.Tensor in [N, H, W] in [-1, 1] scale
    Returns:
        heatmap : torch.Tensor in [N, 3, H, W] in [0, 1]
    """
    x = heatmap_numpy(image.detach().cpu().numpy())
    return torch.from_numpy(x).type_as(image).permute(0, 3, 1, 2)


def visualize_depth(depth, fg_mask=None):
    """Visualize the rendered depth."""
    if fg_mask is not None:
        depth = depth[..., 0]
        fg_depth = depth[fg_mask.bool()]
        zmin, zmax = fg_depth.min(), fg_depth.max()
        depth_disp = (depth - zmin) / (zmax - zmin) * fg_mask
    else:
        zmin, zmax = depth.min(), depth.max()
        depth_disp = (depth - zmin) / (zmax - zmin)
    return depth_disp.repeat(1, 3, 1, 1)


def plot_dict(dic, fpath=None, x=None, N_row=None, N_col=None):
    if N_row is None:
        N = len(dic.keys())
        N_row = math.ceil(math.sqrt(N))
        N_col = math.ceil(N / N_row)

    fig = plt.figure(figsize=(6 * N_col, 4 * N_row))
    for i, (k, v) in enumerate(dic.items()):
        ax = plt.subplot(N_row, N_col, i + 1)
        if type(v) is dict:  # multiple lines with legend
            for iv in v.values():
                if x is not None:
                    ax.plot(x, iv)
                else:
                    ax.plot(iv)
            ax.legend(list(v.keys()))
        else:
            if x is not None:
                ax.plot(x, v)
            else:
                ax.plot(v)
        ax.set_title(k)
    plt.tight_layout()

    if fpath:
        plt.savefig(fpath)
        plt.close()
    return fig


def save_image_grid(img, fname=None, drange=[0, 1], grid_size=[4, 4]):
    """Save a grid of images.
    Args:
        img: Grid of images (np.array) to save.
        fname: File name to save the images.
        drange: Data range.
        grid_size: Size of the image grid.
    Returns:
        numpy image grid.
    """
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo + 1e-9))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        image = Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        image = Image.fromarray(img, 'RGB').save(fname)
    return img


#################################
######## Creating Videos ########
#################################


def get_rotating_angles(
        n_steps=120,
        n_rot=1,
        elev_low=-math.pi/4,
        elev_high=math.pi/4):
    """Return the elevation and azimus angle of 360 rotation.
    """
    half_steps = n_steps // 2
    rot_steps = n_steps // n_rot
    elevs = np.linspace(elev_low, elev_high, half_steps)
    elevs = np.concatenate([elevs, elevs[::-1]])
    elevs = np.concatenate([elevs[half_steps//2:], elevs[:half_steps//2]])
    azims = np.linspace(0, 2 * np.pi, rot_steps)
    azims = np.concatenate([azims, azims[::-1]] * n_rot)
    return azims, elevs


def render_rotate(
        gaussians: List[GaussianParameters],
        bg_color: torch.Tensor,
        fov: float = 0.2443, # 0.2335
        radius: float = 2.7,
        n_frames: int = 60,
        opacity_key='opacity',
        clamp_image=True
    ) -> List[torch.Tensor]:
    """
    Args:
        gaussians: GaussianParameters, the gaussian parameters.
        bg_color: The background color.
        fov: The field of view.
        radius: The radius of the camera.
        n_frames: The number of frames in the video.
    Returns:
        images: List of torch.Tensor, N x (1, 3, H, W), the rendered images, [0, 1]
        depths: List of torch.Tensor, N x (1, 1, H, W), the rendered depths, not normalized.
    """
    images, depths = [], []
    azims, elevs = get_rotating_angles(n_frames)
    device = bg_color.device
    
    n_cam = len(gaussians) if isinstance(gaussians, list) \
        else gaussians.means.shape[0]
    common_kwargs = {
        'fov': fov,
        'radius': radius,
        'device': device
    }

    with torch.no_grad():
        for azim, elev in zip(azims, elevs):
            cams = sample_lookat_camera(azim=azim, elev=elev, **common_kwargs)
            res = splat_gaussians(gaussians, cams * n_cam, bg_color,
                                  opacity_key=opacity_key,
                                  use_gen_size=False)
            if clamp_image:
                res['image'] = res['image'].clamp(0, 1)
            images.append(res['image'].cpu())
            # convert depth into 3 channel images
            depths.append(res['depth'].cpu().repeat(1, 3, 1, 1))
    return images, depths


def write_video(output_path, frames, fps=24):
    """Convert a list of frames to an MP4 video using MoviePy.
    Args:
        output_path: Path to the output video file.
        frames: List of image frames (PIL images or NumPy arrays).
        fps: Frames per second for the output video.
    """
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path,
        codec='libx264', fps=fps,
        preset='ultrafast', threads=1)


def prorcess_depth_frames(depths):
    """Prepare depth frames for visualization.
    """
    depth_min = np.min([d.min() for d in depths])
    depth_max = np.max([d.max() for d in depths])
    norm_fn = lambda x: (x - depth_min) / (depth_max - depth_min)
    for i in range(len(depths)):
        depths[i] = norm_fn(depths[i])


def make_render_rotate_grid(*args):
    """Make a grid of images for the rotating camera.
    """
    frames = []
    for a in args:
        frames = frames + a
    n_frames = len(args[0])
    n_items = len(args)
    # generating permuting indices: avoid using too much contiguous()
    indices = torch.arange(n_items * n_frames).view(n_items, -1)
    indices = indices.permute(1, 0).contiguous().view(-1)
    frames = torch.cat([frames[i] for i in indices])
    frames = frames.view(n_frames, -1, *frames.shape[1:])
    grid_size = int(math.sqrt(frames.shape[1]))
    fn = lambda x: (x.permute(1, 2, 0) * 255).numpy().astype("uint8")
    frames = [fn(vutils.make_grid(f, nrow=grid_size)) for f in frames]
    return frames


def visualize_image_video(
        ggm,
        input_dics : List[TensorDict],
        bg_color: Union[str, torch.Tensor] = 'white',
        n_frames=60,
        render_depth=False):
    """Visualize the generative model by rotating the camera.
    Args:
        ggm: The generative model.
        batch: the input batch of data.
        resolution: The resolution of the rendered image.
        n_frames: The number of frames in the video.
        n_rank: The number of GPUs.
        rank: The rank of the current GPU.
    Returns:
        images: torch.Tensor, (N, 3, H, W), the rendered images.
        depths: torch.Tensor, (N, 1, H, W), the rendered depths.
        alphas: torch.Tensor, (N, 1, H, W), the rendered alphas.
        frames: List of numpy arrays, (N, 3, H, W), the rendered frames.
    """
    n_sample = len(input_dics)
    device = next(ggm.parameters()).device
    if isinstance(bg_color, str):
        bg_color = torch.zeros(3, device=device) if bg_color == 'black' else torch.ones(3, device=device)

    ret_res = {}
    with torch.no_grad():
        ret_dics = [ggm(dic) for dic in input_dics]
        vis_items = []
        if 'canonical' in ret_dics[0]:
            canonical_gaussians = []
            for ret_dic in ret_dics:
                g = ret_dic['canonical']
                if g.xyz.ndim == 4:
                    canonical_gaussians.extend(
                        g.slice_between(0, 1, dim=1).squeeze(dim=1).split())
                elif g.xyz.ndim == 3:
                    canonical_gaussians.extend(
                        g.slice_between(0, 1, dim=0).squeeze(dim=0).split())
            vis_items.append(('canonical', canonical_gaussians))
        for layer_key in ['layer_canonical', 'layer_pixel_aligned', 'layer_early_pa']:
            if layer_key in ret_dics[0]:
                layer_gaussians = []
                for ret_dic in ret_dics:
                    for g in ret_dic[layer_key]:
                        g_ = g.slice_between(0, 1, dim=1).squeeze(dim=1).split()
                        layer_gaussians.extend(g_)
                vis_items.append((layer_key, layer_gaussians))
        if 'pixel_aligned' in ret_dics[0]:
            pa_gaussians = []
            for ret_dic in ret_dics:
                pa_gaussians.extend(ret_dic['pixel_aligned'].rearrange(
                    'b n L ... -> b (n L) ...').split())
            vis_items.append(('pixel_aligned', pa_gaussians))

        for name, Gs in vis_items:
            # retrieve gaussian centers as point cloud
            C0 = 0.28209479177387814
            sh2rgb = lambda x : ((x[..., 0, :3] * C0).clamp(0, 1).cpu().numpy()
                                  * 255).astype(np.uint8)
            n_layer = len(Gs) // n_sample
            #valid_masks = [(Gs[i].opacity[..., 0] > 0.05) for i in range(n_layer)]

            #points = [trimesh.PointCloud(
            #    vertices=Gs[i].xyz[valid_masks[i]].cpu().numpy(),
            #    colors=sh2rgb(Gs[i].sh[valid_masks[i]]))
            #    for i in range(n_layer)]
            points = [trimesh.PointCloud(
                vertices=Gs[i].xyz.cpu().numpy(),
                colors=Gs[i].precomp_color.cpu().numpy() \
                    if Gs[i].sh is None else sh2rgb(Gs[i].sh)
                )
                for i in range(n_layer)]
            # render rotating images
            ims, dps = render_rotate(Gs, bg_color.to(Gs[0].xyz.device), n_frames=n_frames)
            if render_depth:
                prorcess_depth_frames(dps)
                frames = make_render_rotate_grid(ims, dps)
            else:
                frames = make_render_rotate_grid(ims)

            ret_res[name] = frames, points #images, depths, alphas, 
    return ret_res


def save_image_video(prefix, grid_size, images, depths, alphas, frames):
    """Save images and video"""
    #VideoWriter.gen(f'{prefix}_render.mp4', frames)
    write_video(f'{prefix}_render.mp4', frames)
    save_image_grid(images.numpy(),
        f'{prefix}.jpg',
        drange=[0, 1],
        grid_size=grid_size)
    save_image_grid(depths.numpy(),
        f'{prefix}_depth.jpg',
        drange=[depths.min().item(), depths.max().item()],
        grid_size=grid_size)
    save_image_grid(alphas.numpy(),
        f'{prefix}_alpha.jpg',
        drange=[0, 1], grid_size=grid_size)
