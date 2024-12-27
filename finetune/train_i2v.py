import os
import sys
import math
import glob
import torch
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from datetime import timedelta
from typing import List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed, InitProcessGroupKwargs

sys.path.append('.')
from hyvideo.vae import load_vae as load_hyvideo_vae
from hyvideo.modules import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from lib.basetype import EasyDict

import diffusers
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import free_memory
from diffusers.optimization import get_scheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
sys.path.append('.')
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from accelerate.utils import gather_object

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default='lora',
        help=("lora, adalora"),
    )

    # Model information
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default='ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--train_start_step",
        type=int,
        default=0,
        help=("The lowest training step."),
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-controlnet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Dataset information
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        default=None,
        help=("The parquet annotation file for the dataset."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gc_ratio",
        default=1.0,
        type=float,
        help="The ratio of gradient checkpointing layers.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-03, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    return parser.parse_args()


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.scaling_factor = None # need to be filled later
        # find all video latents
        self.video_latent_files = sorted(glob.glob(os.path.join(root_dir, '*.pth')))

    def __len__(self):
        return len(self.video_latent_files)

    def __getitem__(self, index):
        latent_path = self.video_latent_files[index]
        idx = latent_path[latent_path.rfind('/')+1:]
        idx = int(idx[:idx.find('.')])
        dic = torch.load(latent_path)
        video_latents = dic['video_latent'][0] * self.scaling_factor
        first_latents = video_latents.clone()
        first_latents[:, 1:] = 0

        return {
            'prompt_embed1': dic['prompt_embed1'][0],
            'prompt_embed2': dic['prompt_embed2'][0],
            'attention_mask1': dic['attention_mask1'][0],
            'video_latents': video_latents,
            'image_latents': first_latents,
        }


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


@torch.no_grad
def visualize_latent(vae, cond_latents, save_path):
    x = cond_latents.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor
    cond_image = vae.decode(x).sample
    cond_image = cond_image[0].permute(1, 2, 3, 0).float().clamp(-1, 1)
    cond_image_np = (cond_image * 127.5 + 127.5).cpu().numpy().astype('uint8')
    proc = lambda x : [Image.fromarray(a[0]) for a in np.split(cond_image_np, x.shape[0])]
    cond_image = proc(cond_image)
    export_to_video(cond_image, save_path)


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str = 'sigma_sqrt', sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def get_rotary_pos_embed(model, video_length=65, height=960, width=960, rope_theta=256):
    target_ndim = 3
    ndim = 5 - 2
    latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]

    if isinstance(model.patch_size, int):
        assert all(s % model.patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // model.patch_size for s in latents_size]
    elif isinstance(model.patch_size, list):
        assert all(
            s % model.patch_size[idx] == 0
            for idx, s in enumerate(latents_size)
        ), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [
            s // model.patch_size[idx] for idx, s in enumerate(latents_size)
        ]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = model.hidden_size // model.heads_num
    rope_dim_list = model.rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=rope_theta,
        use_real=True,
        theta_rescale_factor=1,
    )

    return freqs_cos, freqs_sin




def inference_step(input_dic, transformer, scheduler):
    # Add noise to the model input according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    video_latents = input_dic['video_latents']
    device, dtype = video_latents.device, video_latents.dtype
    n_dim = video_latents.ndim

    embedded_guidance_scale = 6.0
    guidance_expand = torch.tensor(
        [embedded_guidance_scale] * video_latents.shape[0],
        dtype=dtype, device=device) * 1000.0
    #guidance_expand = None

    timesteps = input_dic['timesteps']
    t_expand = timesteps.repeat(video_latents.shape[0])

    schedule_timesteps = scheduler.timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)

    # Add noise according to flow matching.
    noise = torch.randn_like(video_latents, device=device, dtype=dtype)
    noisy_model_input = (1.0 - sigma) * video_latents + sigma * noise

    # Predict the noise residual
    noise_pred = transformer(  # For an input image (129, 192, 336) (1, 256, 256)
        noisy_model_input,  # [2, 16, 33, 24, 42]
        t_expand,  # [2]
        text_states=input_dic['prompt_embed1'],  # [2, 256, 4096]
        text_mask=input_dic['attention_mask1'],  # [2, 256]
        text_states_2=input_dic['prompt_embed2'],  # [2, 768]
        freqs_cos=input_dic['freqs_cis'][0],  # [seqlen, head_dim]
        freqs_sin=input_dic['freqs_cis'][1],  # [seqlen, head_dim]
        guidance=guidance_expand,
        return_dict=True,
    )['x']
    z0 = noisy_model_input + noise_pred * sigma
    return z0


if __name__ == "__main__":
    args = get_args()

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=3600))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )
    device = accelerator.device
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    scheduler = FlowMatchDiscreteScheduler(
        shift=7.0,
        reverse=False, # in inference, this is True.
        solver='euler'
    )

    transformer = HYVideoDiffusionTransformer(
        EasyDict(text_states_dim=4096, text_states_dim_2=768),
        in_channels=16,
        out_channels=16,
        device=device, dtype=dtype,
        **HUNYUAN_VIDEO_CONFIG['HYVideo-T/2-cfgdistill'],
    )
    sd = torch.load(args.pretrained_model_path, map_location='cpu')
    transformer.load_state_dict(sd['module'], strict=True)
    transformer.enable_gradient_checkpointing()

    vae, _, s_ratio, t_ratio = load_hyvideo_vae(
        '884-16c-hy', 'fp16',
        device=device,
    )
    vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

    # We only train the additional adapter controlnet layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to('cpu', dtype=dtype)
    transformer.to(device, dtype=dtype)

    # now we will add new LoRA weights to the attention layers
    if args.adapter == 'lora':
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights=True,
            # txt_attn_proj, img_attn_proj: out
            # single stream block: "linear1", "linear2"
            target_modules=['txt_attn_proj', 'img_attn_proj', 'txt_attn_qkv', 'img_attn_qkv'],
        )
        transformer.add_adapter(transformer_lora_config)


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    params_to_optimize = [{
            "params": transformer_lora_parameters,
            "lr": args.learning_rate
        }]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    train_dataset = VideoDataset(
        root_dir=args.data_root,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers('hunyuan-lora', config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f" Sampling training start step: {args.train_start_step}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    train_dataset.scaling_factor = vae.config.scaling_factor

    n_bucs = min(8, accelerator.num_processes * accelerator.gradient_accumulation_steps)
    device = accelerator.device
    rng = np.random.RandomState(2024 + accelerator.local_process_index)
    freqs_cis = get_rotary_pos_embed(transformer)
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            rank = accelerator.local_process_index
            accum = accelerator.gradient_accumulation_steps
            buc_id = (rank * accum + step % accum) % n_bucs

            with accelerator.accumulate(models_to_accumulate):
                video_latents = batch["video_latents"].to(dtype=dtype).to(device)

                u = compute_density_for_timestep_sampling(
                    weighting_scheme='logit_normal',
                    batch_size=video_latents.shape[0],
                    logit_mean=1.0,
                    logit_std=1.0,
                    mode_scale=1.0,
                )
                indices = (u * scheduler.config.num_train_timesteps).long()
                timesteps = scheduler.timesteps[indices].to(device=device)
                timesteps = timesteps.to(device)

                input_dic = dict(
                    video_latents=video_latents,
                    timesteps=timesteps,
                    freqs_cis=freqs_cis,
                    prompt_embed1=batch["prompt_embed1"].to(dtype=dtype).to(device),
                    prompt_embed2=batch["prompt_embed2"].to(dtype=dtype).to(device),
                    attention_mask1=batch["attention_mask1"].to(device),
                    image_latents=batch["image_latents"].to(device),
                )

                z0 = inference_step(input_dic, transformer, scheduler)

                weights = compute_loss_weighting_for_sd3(
                    weighting_scheme='sigma_sqrt',
                    sigmas=scheduler.sigmas,
                )
                while len(weights.shape) < len(z0.shape):
                    weights = weights.unsqueeze(-1)
                dsm_loss = weights * (z0 - video_latents) ** 2

                loss = dsm_loss.mean()
                accelerator.backward(loss)

                with torch.no_grad():
                    parameters = transformer_lora_parameters
                    grad_max = [p.grad.abs().max() for p in parameters if p.grad is not None]
                    grad_max = float(sum(grad_max) / len(grad_max)) if grad_max else 0.0

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 1 or global_step == args.max_train_steps:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")

                        model = unwrap_model(transformer)
                        transformer_lora_layers = get_peft_model_state_dict(model)

                        HunyuanVideoPipeline.save_lora_weights(
                            save_directory=args.output_dir,
                            weight_name=f'pytorch_lora_weights_{global_step}.safetensors',
                            transformer_lora_layers=transformer_lora_layers)

                if False:#global_step % 100 == 0:
                    print("Visualize latent")
                    
                
                vae.to(device); transformer.to('cpu'); controlnet.to('cpu'); free_memory()
                visualize_latent(vae, cond_latents.bfloat16(), f'{args.output_dir}/condviz_{global_step}_{rank}.mp4')
                visualize_latent(vae, video_latents.bfloat16(), f'{args.output_dir}/videoviz_{global_step}_{rank}.mp4')
                vae.to('cpu'); transformer.to(device); controlnet.to(device); free_memory()

                if global_step % 500 == 0:
                    print("Visualize latent series")
                with torch.no_grad():
                    for t in [980, 999]:#[999, 950, 900, 850, 800]:
                        t = torch.tensor([t]).long().to(device)
                        vae.to('cpu'); transformer.to(device); controlnet.to(device); free_memory()
                        model_pred = inference_step(time_embed_fn, args, video_latents, cond_latents, image_latents, t, transformer, controlnet, scheduler, collect_layers, prompt_embeds, image_rotary_emb, rng)
                        vae.to(device); transformer.to('cpu'); controlnet.to('cpu'); free_memory()
                        visualize_latent(vae, model_pred.bfloat16(), f'{args.output_dir}/modelpred_{global_step}_{rank}_sample{int(t[0])}.mp4')
                vae.to('cpu'); transformer.to(device); controlnet.to(device); free_memory()

                if step == 4:
                    exit(0)

                if False: #global_step % 1000 == 0: # run full pipeline
                    pipe = CogVideoXControlNetPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=text_encoder,
                        transformer=transformer,
                        controlnet=unwrap_model(controlnet),
                        scheduler=scheduler,
                        torch_dtype=weight_dtype).to(device)
                    pipe.i2v_backbone = '5b' in args.pretrained_model_name_or_path

                    first_latent = image_latents[:, :1] / pipe.vae_scaling_factor_image
                    cond_latent = cond_latents / pipe.vae_scaling_factor_image
                    video_generate = pipe(
                        prompt_embeds=prompt_embeds,
                        first_frame_or_latents=first_latent,
                        cond_video_or_latents=cond_latent,
                        #latents=video_latents,
                        num_inference_steps=50,
                        #strength=0.95,
                        control_start_ratio=0.0, control_end_ratio=0.5,
                        use_dynamic_cfg=True, guidance_scale=2,
                        generator=torch.Generator().manual_seed(1),
                        save_inter_video_path=args.output_dir + f'/{global_step}_{rank}_'
                    ).frames[0]
                    del pipe
                    vae.to('cpu'); transformer.to(device); controlnet.to(device); text_encoder.to('cpu'); free_memory()

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "grad_max": grad_max,
            }
            bin_loss = gather_object((int(timesteps[0]), dsm_loss.mean().detach().item()))
            high_noise_loss = []
            for i in range(len(bin_loss) // 2):
                t, val = bin_loss[i * 2], bin_loss[i * 2 + 1]
                if t >= 900:
                    high_noise_loss.append(val)
            logs["high_noise_loss"] = sum(high_noise_loss) / len(high_noise_loss) if high_noise_loss else 0
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
