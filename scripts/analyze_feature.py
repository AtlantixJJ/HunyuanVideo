import os
import sys
import torch
from pathlib import Path
sys.path.append('.')
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
import matplotlib.pyplot as plt


def generate_feature():
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
        collect_feature=True
    )

    features = outputs['features']
    sample = outputs['samples']
    os.makedirs("expr/analysis", exist_ok=True)

    torch.save(features, f"expr/analysis/features_seed{args.seed}.pt")
    save_videos_grid(sample[None], f"expr/analysis/output_seed{args.seed}.mp4", fps=24)
    return sample, features


if __name__ == "__main__":
    sample, features = generate_feature()

    # layer statistics
    os.makedirs("expr/analysis/layer_channel", exist_ok=True)
    L, H, W = 5, 90 // 2, 90 // 2
    image_length = features[0][0][0].shape[1]
    #for layer_features in features:
    layer_features = features[5]
    for i, res in enumerate(layer_features):
        print(f"visualizing layer {i}")
        if len(res) == 2:
            img, txt = res
            #img.view(L, H, W, -1) # (L, H, W)
        else:
            img, txt = res[:, :image_length], res[:, image_length:]
        img, txt = img.float().cuda(), txt.float().cuda()
        channel_mean = img[0].mean(0).cpu()
        max_channel_indices = [str(int(x)) for x in channel_mean.abs().argsort()[-3:]]
        s = ','.join(max_channel_indices)
        channel_dev = img[0].std(0).cpu()
        txt_mean = txt[0].mean(0).cpu()
        txt_dev = txt[0].std(0).cpu()
        ax = plt.subplot(2, 2, 1)
        ax.plot(channel_mean)
        ax.set_title(f"Image {i} Mean")
        ax = plt.subplot(2, 2, 2)
        ax.plot(channel_dev)
        ax.set_title(f"Image {i} Deviation")
        ax = plt.subplot(2, 2, 3)
        ax.plot(txt_mean)
        ax.set_title(f"Text {i} Mean")
        ax = plt.subplot(2, 2, 4)
        ax.plot(txt_dev)
        ax.set_title(f"Text {i} Deviation")
        plt.savefig(f"expr/analysis/layer_channel/layer{i:02d}_{s}.png")
        plt.close()

    # max channel visualization
    max_idx = 1028
    os.makedirs("expr/analysis/channel_1028", exist_ok=True)
    norm_fn = lambda x: (x - x.min()) / (x.max() - x.min())
    for i in range(len(features[0])): # layer idx
        print(f"visualizing layer {i}")
        x = []
        for layer_features in features: # time step 
            res = layer_features[i]
            if len(res) == 2:
                img, txt = res
                #img.view(L, H, W, -1) # (L, H, W)
            else:
                img, txt = res[:, :image_length], res[:, image_length:]
            x.append(img[0, :, max_idx].float())
        x = torch.stack(x).view(-1, L * H, W) # (T, LHW) -> (T, LH, W)
        x = x.permute(1, 0, 2).reshape(L * H, -1) # (LH, W, T) -> (LH, TW)
        disp = norm_fn(x)
        plt.imshow(disp.cpu().numpy())
        plt.savefig(f"expr/analysis/channel_1028/layer{i:02d}.png")
        plt.close()

