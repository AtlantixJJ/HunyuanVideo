import torch
import sys
import diffusers
import glob
import numpy as np
from PIL import Image
from diffusers.utils import load_video, export_to_video
from tqdm import tqdm
sys.path.append('.')
from hyvideo.constants import PROMPT_TEMPLATE
from hyvideo.text_encoder import TextEncoder
from hyvideo.vae import load_vae

device = 'cuda'
data_dir = 'data/hunyuan_distillation_cfg6'
video_files = sorted(glob.glob(f'{data_dir}/*.mp4'))
prompt_file = f'{data_dir}/new_prompts.txt'
prompts = [l.strip() for l in open(prompt_file, 'r').readlines() if len(l) > 5]
dtype = torch.half
torch.set_grad_enabled(False)


def preprocess(images):
    img = np.stack([np.asarray(img) for img in images]) 
    img = torch.from_numpy(img).float().div(255.) * 2 - 1 # (N, H, W, 3)
    return img.permute(3, 0, 1, 2).unsqueeze(0) # (1, 3, N, H, W)


def postprocess(images):
    # images shape: [1, 3, 65, 960, 960]
    images = ((images[0].permute(1, 2, 3, 0) + 1) / 2 * 255).clamp(0, 255)
    images = images.byte().cpu().numpy()
    return [Image.fromarray(img) for img in images]


def encode_prompt(prompt, text_encoder, device, clip_skip=None):
    """Modified from hyvideo.diffusion.pipelines"""

    num_videos_per_prompt = 1
    data_type = 'image'

    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

    if clip_skip is None:
        prompt_outputs = text_encoder.encode(
            text_inputs, data_type=data_type, device=device
        )
        prompt_embeds = prompt_outputs.hidden_state
    else:
        prompt_outputs = text_encoder.encode(
            text_inputs,
            output_hidden_states=True,
            data_type=data_type,
            device=device,
        )
        # Access the `hidden_states` first, that contains a tuple of
        # all the hidden states from the encoder layers. Then index into
        # the tuple to access the hidden states from the desired layer.
        prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
        # We also need to apply the final LayerNorm here to not mess with the
        # representations. The `last_hidden_states` that we typically use for
        # obtaining the final prompt representations passes through the LayerNorm
        # layer.
        prompt_embeds = text_encoder.model.text_model.final_layer_norm(
            prompt_embeds
        )

    attention_mask = prompt_outputs.attention_mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        bs_embed, seq_len = attention_mask.shape
        attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
        attention_mask = attention_mask.view(
            bs_embed * num_videos_per_prompt, seq_len
        )

    prompt_embeds_dtype = text_encoder.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    if prompt_embeds.ndim == 2:
        bs_embed, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
    else:
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

    return prompt_embeds, attention_mask


crop_start = PROMPT_TEMPLATE['dit-llm-encode']['crop_start']
text_encoder = TextEncoder(
    text_encoder_type='llm',
    max_length=256 + crop_start,
    text_encoder_precision='fp16',
    tokenizer_type='llm',
    prompt_template=PROMPT_TEMPLATE['dit-llm-encode'],
    prompt_template_video=PROMPT_TEMPLATE['dit-llm-encode-video'],
    hidden_state_skip_layer=2,
    apply_final_norm=False,
    reproduce=True,
    device=device,
)

text_encoder_2 = TextEncoder(
    text_encoder_type='clipL',
    max_length=77,
    text_encoder_precision='fp16',
    tokenizer_type='clipL',
    reproduce=True,
    device=device,
)
vae, _, s_ratio, t_ratio = load_vae(
    '884-16c-hy',
    'fp16',
    device=device
)
vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
vae.enable_tiling()

prompt_anns = []

print("Preprocessing prompts...")
for i, prompt in enumerate(tqdm(prompts)):
    prompt_embed1, attention_mask1 = encode_prompt(prompt, text_encoder, device)
    prompt_embed2, attention_mask2 = encode_prompt(prompt, text_encoder_2, device)
    prompt_anns.append([prompt_embed1.cpu(), attention_mask1.cpu(), prompt_embed2.cpu()])
torch.save(prompt_anns, f'{data_dir}/new_prompts.pth')

print("Preprocessing video...")
for video_file in tqdm(video_files):
    images = preprocess(load_video(video_file)).to(device).half()
    video_latent = vae.encode(images).latent_dist.sample()

    #image_recon = vae.decode(video_latent).sample
    #export_to_video(postprocess(image_recon), 'test.mp4')
    #from IPython import embed; embed()

    torch.save(video_latent.cpu(), video_file.replace(".mp4", ".pth"))
