import torch
import sys
import os, glob
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video, load_image
from diffusers import AutoencoderKLHunyuanVideo
sys.path.insert(0, '.')


def crop_resize(pil_image, size=(720, 720)):
    old_size = pil_image.size
    min_size = min(old_size)
    new_size = (min_size, min_size)
    # crop with new size
    pil_image = pil_image.crop(((old_size[0] - new_size[0]) // 2,
                                 (old_size[1] - new_size[1]) // 2,
                                 (old_size[0] + new_size[0]) // 2,
                                 (old_size[1] + new_size[1]) // 2))
    # resize to desired size
    return pil_image.resize(size)


def preprocess(images):
    img = np.stack([np.asarray(img) for img in images]) 
    img = torch.from_numpy(img).float().div(255.) * 2 - 1 # (N, H, W, 3)
    return img.permute(3, 0, 1, 2).unsqueeze(0) # (1, 3, N, H, W)


def postprocess(images):
    # images shape: [1, 3, 65, 960, 960]
    images = ((images[0].permute(1, 2, 3, 0) + 1) / 2 * 255).clamp(0, 255)
    images = images.byte().cpu().numpy()
    return [Image.fromarray(img) for img in images]


torch.set_grad_enabled(False)
root_dir = 'data/NeRSemble'
device = 'cuda'
instance_dirs = sorted([d for d in glob.glob(f'{root_dir}/data/*')
                        if os.path.isdir(d)])
vae = AutoencoderKLHunyuanVideo.from_pretrained(
    'hunyuanvideo-community/HunyuanVideo',
    subfolder='vae', torch_dtype=torch.float16).to(device)
vae.enable_tiling()

with open(f'{root_dir}/camera_order.txt', 'r') as f:
    sorted_names = f.read().splitlines()

cname_fmt = 'data/NeRSemble/data/017/extra_sequences/EMO-1-shout+laugh/frame_00000/images-2fps/cam_{}.jpg'
images = [crop_resize(load_image(cname_fmt.format(cam_name))) for cam_name in sorted_names]
x = preprocess(images).to(device=device, dtype=torch.float16)
z = vae.encode(x).latent_dist.sample()
x_rec = vae.decode(z).sample

# concatenate images with rec_images in the columns
rec_images = postprocess(x_rec)
disp = [Image.fromarray(np.concatenate([np.asarray(x), np.asarray(y)], axis=1))
        for x, y in zip(images, rec_images)]
export_to_video(disp, 'expr/nersemble_vae_recon.mp4')
