import sys
import argparse
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video, load_video, load_image

sys.path.append('.')
from lib.pipeline_hunyuan_image2video import HunyuanImageToVideoPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Sample diffusers")
    # model loading
    parser.add_argument("--model_name", type=str, default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--lora_path", type=str, default="expr/lora/pytorch_lora_weights_50.safetensors")
    # generation mode
    parser.add_argument("--mode", type=str, default="i2v", choices=["t2v", "i2v"])
    # input
    parser.add_argument("--first_image_path", type=str, default="data/hunyuan_distillation_cfg6/00000_seed1981.mp4")
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic")
    # video size
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=17)
    # solver
    parser.add_argument("--num_infer_steps", type=int, default=50)
    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="expr/test/lora.mp4")
    parser.add_argument("--fps", type=int, default=24)

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    device = 'cuda'

    # has to initialize transformer here using bfloat16, otherwise the model will explode
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_name, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    if args.mode == 't2v':
        pipe = HunyuanVideoPipeline.from_pretrained(
            args.model_name,
            transformer=transformer,
            torch_dtype=torch.float16)
    elif args.mode == 'i2v':
        pipe = HunyuanImageToVideoPipeline.from_pretrained(
            args.model_name,
            transformer=transformer,
            torch_dtype=torch.float16)
        pipe.transformer.rope.set_mode('i2v')
    
    if len(args.first_image_path) > 0:
        # if the path is a video file
        if args.first_image_path.endswith('.mp4'):
            first_frame = load_video(args.first_image_path)[0]
        else:
            first_frame = load_image(args.first_image_path)

    pipe.vae.enable_tiling()
    pipe.to(device)

    if len(args.lora_path) > 0:
        print(f"Loading LoRA weights from {args.lora_path}")
        pipe.load_lora_weights(args.lora_path, adapter_name="adapter")
        pipe.fuse_lora(['transformer'], 1.0)

    common_dict = dict(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_infer_steps,
    )

    if args.mode == 't2v':
        output = pipe(**common_dict).frames[0]
    elif args.mode == 'i2v':
        output = pipe(
            first_frame_or_latents=first_frame,
            save_intermediate_dir=args.output_path[:args.output_path.rfind('.')],
            **common_dict
        ).frames[0]

    export_to_video(output, args.output_path, fps=args.fps)