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

import transformers
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

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
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Path to pretrained lora",
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--train_start_step",
        type=int,
        default=0,
        help=("The lowest training step."),
    )
    parser.add_argument(
        "--stochastic_layers",
        type=int,
        default=0,
        help=("Whether to use stochastic layers in the controlnet."),
    )
    parser.add_argument(
        "--controlnet_mode",
        type=str,
        default='elementwise',
        required=True,
        help=("Controlnet type. (canny, hed, etc.)"),
    )
    parser.add_argument(
        "--fixed_cond_timestep",
        type=int,
        default=-1,
        help="Whether to use fixed timestep for conditioning.",
    )
    parser.add_argument(
        "--pretrained_controlnet_path",
        type=str,
        default=None,
        required=False,
        help=("Path to controlnet .pt checkpoint."),
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
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
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )

    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
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
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
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

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
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
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--nccl_timeout", type=int, default=600, help="NCCL backend timeout in seconds.")
    return parser.parse_args()


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        prompt_embed_file: Optional[str] = None, # pre-computed prompt embeddings
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.prompt_embed_dic = torch.load(prompt_embed_file)
        self.scaling_factor = None # need to be filled later
        # find all video latents
        self.video_latent_files = sorted(glob.glob(os.path.join(root_dir, '*cogvideoxlatents.pth')))

    def __len__(self):
        return len(self.video_latent_files)

    def __getitem__(self, index):
        latent_path = self.video_latent_files[index]
        idx = latent_path[latent_path.rfind('/')+1:]
        idx = int(idx[:idx.find('_')])
        print(latent_path, idx)
        vl_params = torch.load(latent_path)
        latent_dist = DiagonalGaussianDistribution(vl_params)
        video_latents = latent_dist.sample()[0] * self.scaling_factor
        first_latents = latent_dist.sample()[0] * self.scaling_factor
        first_latents[:, 1:] = 0

        return {
            "prompt_embeds": self.prompt_embed_dic[idx][0], # remove batch dimension
            "video_latents": video_latents,
            "image_latents": first_latents,
        }


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


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


class CNN(torch.nn.Module):
    def __init__(self, in_dim=16, hidden_dim=256, out_dim=8, n_layer=3):
        super().__init__()
        
        dims = [in_dim] + [hidden_dim] * (n_layer - 2) + [out_dim]
        self.conv_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(prev_dim, cur_dim, 3, 1, 1)
            for prev_dim, cur_dim in zip(dims[:-1], dims[1:])])

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, input):
        x = input
        for layer in self.conv_layers:
            x = torch.nn.functional.relu(layer(x))
        return x + input[:, :self.out_dim]


def inference_step(args, video_latents, image_latents, timesteps, transformer, scheduler, prompt_embeds, image_rotary_emb):
    # Add noise to the model input according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noise = torch.randn_like(video_latents)
    noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
    noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)

    # Predict the noise residual
    model_output = transformer(
        hidden_states=noisy_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
        gc_ratio=args.gc_ratio,
    )[0]

    model_pred = scheduler.get_velocity(
        model_output, noisy_video_latents, timesteps)
    return model_pred


if __name__ == "__main__":
    args = get_args()

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

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
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    num_collect = args.stochastic_layers
    num_layers = len(transformer.transformer_blocks)
    collect_layers = [2, 3, 8, 9, 11, 15, 21, 33] # hard fixed
    controlnet_config = dict(
        num_attention_heads=48 if "5b" in args.pretrained_model_name_or_path.lower() else 30,
        attention_head_dim=64,
        mode=args.controlnet_mode,
        stochastic_layers=args.stochastic_layers,
        collect_layers=collect_layers,
        fixed_condtime=args.fixed_cond_timestep,
        timemed_dim=512,
        bottleneck_dim=128,
        num_layers=num_layers,
    )
    controlnet = CogVideoXSimpleControlnet(controlnet_config)


    if args.pretrained_controlnet_path:
        sd = torch.load(args.pretrained_controlnet_path, map_location='cpu')
        m, u = controlnet.load_state_dict(sd['state_dict'], strict=False)
        print(f'[ Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # We only train the additional adapter controlnet layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.requires_grad_(True)

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

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to('cpu', dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    print("IN DEBUG MODE")
    torch.set_grad_enabled(False)


    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    if args.adapter == 'lora':
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    elif args.adapter == 'adalora':
        transformer_lora_config = AdaLoraConfig(
            #r=args.rank,
            target_r=args.lora_rank,
            init_r=args.lora_rank * 2,
            lora_alpha=args.lora_rank,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            peft_type="ADALORA",
        )
    transformer.add_adapter(transformer_lora_config)

    # pipe here is only used to load the weights
    if args.pretrained_lora_path:
        print(f'[ Loading pretrained LoRA weights from {args.pretrained_lora_path} ]')
        pipe = CogVideoXControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            controlnet=controlnet,
            scheduler=scheduler,
            torch_dtype=weight_dtype).to(accelerator.device)
        pipe.i2v_backbone = '5b' in args.pretrained_model_name_or_path
        pipe.load_lora_weights(args.pretrained_lora_path, adapter_name="adapter")
        pipe.fuse_lora(lora_scale=1.0, components=["transformer"])
        del pipe


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
    trainable_parameters = list(filter(lambda p: p.requires_grad, controlnet.parameters()))

    params = controlnet.out_projectors
    # Optimization parameters
    params_to_optimize = [{
            "params": params if controlnet.mode == "elementwise" \
                else params.parameters(),
            "lr": args.learning_rate
        }, {
            "params": controlnet.weights,
            "lr": 0.01
        }, {
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
        ann_file=args.ann_file,
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
    transformer, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, controlnet, optimizer, train_dataloader, lr_scheduler
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
        tracker_name = args.tracker_name or "cogvideox-controlnet-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

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

    train_dataset.prompt_embed = compute_prompt_embeddings(
        tokenizer,
        text_encoder,
        train_dataset.prompt_template,
        model_config.max_text_seq_length,
        accelerator.device,
        weight_dtype,
        requires_grad=False,
    )
    train_dataset.scaling_factor = vae.config.scaling_factor
    tokenizer = tokenizer
    text_encoder = text_encoder.cpu()

    transformer_ = unwrap_model(transformer)
    time_embed_fn = lambda x: transformer_.time_embedding(
        transformer_.time_proj(x).bfloat16(), None)
    n_bucs = min(8, accelerator.num_processes * accelerator.gradient_accumulation_steps)
    min_train_steps = args.train_start_step
    buc_size = float(scheduler.config.num_train_timesteps - min_train_steps) / n_bucs
    full_layer_indices = list(range(num_layers))
    device = accelerator.device
    rng = np.random.RandomState(2024 + accelerator.local_process_index)
    # Prepare rotary embeds
    image_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=args.height,
            width=args.width,
            num_frames=13, # hard code
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_size=model_config.patch_size,
            attention_head_dim=model_config.attention_head_dim,
            device=accelerator.device,
        )
        if model_config.use_rotary_positional_embeddings
        else None
    )
    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [controlnet, transformer]
            rank = accelerator.local_process_index
            accum = accelerator.gradient_accumulation_steps
            buc_id = (rank * accum + step % accum) % n_bucs

            with accelerator.accumulate(models_to_accumulate):
                video_latents = batch["video_latents"].permute(0, 2, 1, 3, 4).to(dtype=weight_dtype).to(device)
                prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype).to(device) 
                image_latents = batch["image_latents"].permute(0, 2, 1, 3, 4).to(device)
                cond_latents = batch["normal_latents"].permute(0, 2, 1, 3, 4).to(device)

                batch_size = video_latents.shape[0]
                t = torch.rand((batch_size,), device=video_latents.device)
                timesteps = (buc_size * t + buc_id * buc_size).long() + min_train_steps
                if rank == 0: # set rank 0 to train before min_train_steps
                    t = torch.rand((batch_size,), device=video_latents.device)
                    timesteps = ((buc_size + min_train_steps) * t).long()
                tar_timembed = time_embed_fn(timesteps)

                model_pred = inference_step(time_embed_fn, args, video_latents, cond_latents, image_latents, timesteps, transformer, controlnet, scheduler, collect_layers, prompt_embeds, image_rotary_emb, rng)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)
                dsm_loss = weights * (model_pred - video_latents) ** 2

                loss = dsm_loss.mean()
                #accelerator.backward(loss)

                with torch.no_grad():
                    parameters = [p for p in controlnet.parameters()]
                    grad_max = [p.grad.abs().max() for p in parameters if p.grad is not None]
                    grad_max = float(sum(grad_max) / len(grad_max)) if grad_max else 0.0

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        controlnet.parameters(), args.max_grad_norm)
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm)

                #if accelerator.state.deepspeed_plugin is None:
                #    optimizer.step()
                #    optimizer.zero_grad()

                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 1 or global_step == args.max_train_steps:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                        model = unwrap_model(controlnet)
                        torch.save({
                            'state_dict': model.state_dict(),
                            'config': model.config_dict()
                            }, save_path)
                        logger.info(f"Saved state to {save_path}")
                        model = unwrap_model(transformer)
                        transformer_lora_layers = get_peft_model_state_dict(model)

                        CogVideoXControlNetPipeline.save_lora_weights(
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
