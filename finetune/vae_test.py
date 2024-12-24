import torch
import diffusers
import numpy as np
from diffusers.utils import load_video

from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hyvideo.vae import load_vae

device = 'cuda'
video_file = 'results/2024-12-22-23:21:44_seed42_A cat walks on the grass, realistic style..mp4'
torch.set_grad_enabled(False)

def preprocess(images):
    img = np.stack([np.asarray(img) for img in images]) 
    img = torch.from_numpy(img).float().div(255.) * 2 - 1 # (N, H, W, 3)
    return img.permute(3, 0, 1, 2).unsqueeze(0) # (1, 3, N, H, W)


images = preprocess(load_video(video_file)).to(device).half()

vae, _, s_ratio, t_ratio = load_vae(
    '884-16c-hy',
    'fp16',
    device=device
)
vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
vae.enable_tiling()
res = vae.encode(images)