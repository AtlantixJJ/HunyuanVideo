import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import numpy as np
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

    data_dir = 'data/hunyuan_distillation'
    prompt_file = f'{data_dir}/new_prompts.txt'
    prompts = [l.strip() for l in open(prompt_file, 'r').readlines() if len(l) > 5]
    #n_rank, rank = int(args.n_rank), int(args.rank)
    n_rank, rank = 1, 0
    prompts = [(idx, prompts[idx]) for idx in range(rank, len(prompts), n_rank)]
    for idx, prompt in prompts:
        # Start sampling
        output_path = f'{data_dir}/{idx:05d}.mp4'
        print(prompt)
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt, 
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
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        samples = outputs['samples']

        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                save_videos_grid(sample, output_path, fps=24)
                logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()
