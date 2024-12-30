import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.to("cuda")

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=720,
    width=720,
    num_frames=17,
    num_inference_steps=50,
).frames[0]
export_to_video(output, "output.mp4", fps=15)