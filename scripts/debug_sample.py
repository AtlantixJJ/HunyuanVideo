import os
import sys
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
sys.path.append('.')
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    fs, ts = (args.rank > 0), (args.n_rank > 0) # a hack here

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        do_frame_scaling=fs,
        do_theta_scaling=ts,
        save_intermediate_dir=f'expr/test/fs{fs}_ts{ts}_'
    )
    samples = outputs['samples']

    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        save_path = f"expr/test/fs{fs}_ts{ts}.mp4"
        save_videos_grid(samples[None], save_path, fps=24)
        logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()