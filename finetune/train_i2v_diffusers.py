#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import sys
import argparse
import copy
import logging
import os
#os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
#os.environ['NCCL_DEBUG'] = 'DETAIL'
import glob
from datetime import timedelta
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, DistributedDataParallelKwargs, ProjectConfiguration, set_seed, InitProcessGroupKwargs
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from accelerate.utils import gather_object
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline, AutoencoderKLHunyuanVideo
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import check_min_version, is_wandb_available, load_image, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
sys.path.append('.')
from lib.transformer_hunyuan_video import MyHunyuanVideoTransformer3DModel
from PIL import Image
from diffusers.utils import export_to_video


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)

NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def log_validation(transformer, args, accelerator, weight_dtype, step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation:
        transformer = accelerator.unwrap_model(transformer)
        pipeline = FluxControlPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
        )
        initial_channels = transformer.config.in_channels
        pipeline = FluxControlPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )
        pipeline.load_lora_weights(args.output_dir)
        assert (
            pipeline.transformer.config.in_channels == initial_channels * 2
        ), f"{pipeline.transformer.config.in_channels=}"

    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device.type, weight_dtype)

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = load_image(validation_image)
        # maybe need to inference on 1024 to get a good image
        validation_image = validation_image.resize((args.resolution, args.resolution))

        images = []

        for _ in range(args.num_validation_images):
            with autocast_ctx:
                # need to fix in pipeline_flux_controlnet
                image = pipeline(
                    prompt=validation_prompt,
                    control_image=validation_image,
                    num_inference_steps=50,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    max_sequence_length=512,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]
            image = image.resize((args.resolution, args.resolution))
            images.append(image)
        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                formatted_images = []
                formatted_images.append(np.asarray(validation_image))
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")

        elif tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        free_memory()
        return image_logs


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-lora-{repo_id}

These are Control LoRA weights trained on {base_model} with new type of conditioning.
{img_str}

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "flux",
        "flux-diffusers",
        "text-to-image",
        "diffusers",
        "control-lora",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Control LoRA training script.")

    # model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="the guidance scale used for transformer.",
    )
    parser.add_argument(
        "--embed_cfg_scale",
        type=float,
        default=6.0,
        help="the guidance scale for embedding.",
    )

    # position embedding
    parser.add_argument(
        "--rotary_mode",
        type=str,
        default='i2v-temporal-1',
        help=("i2v-temporal-2, i2v-temporal-1, i2v-spatial"),
    )

    # lora
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_lora_bias",
        action="store_true",
        help="If training the bias of lora_B layers."
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )
    parser.add_argument(
        "--gaussian_init_lora",
        action="store_true",
        help="If using the Gaussian init strategy. When False, we follow the original LoRA init strategy.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default='',
        help=(
            'The path to lora weights to start finetuning.'
        ),
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default='',
        help=(
            'The path to weights to start finetuning.'
        ),
    )
    parser.add_argument(
        "--vpt_mode",
        type=str,
        default='deep-add-1',
        help="Arguments for VPT mode.",
    )

    # dataset
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="The directory where the video latents are stored.",
    )
    parser.add_argument(
        "--prompt_embed_ann",
        type=str,
        default=None,
        required=True,
        help="The path to precomputed prompt embeddings (to avoid cpuoffloading text encoder frequently)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=720,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    # training hyperparameters
    # flow matching scheduler control
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    # training control
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="hunyuan_train_lora",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )

    parser.add_argument(
        "--train_norm_layers",
        action="store_true",
        help="Whether to train the norm scales."
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
    #gc_ratio
    parser.add_argument(
        "--gc_ratio",
        type=float,
        default=1.0,
        help="The ratio of (double block) layers to use gradient checkpointing."
    )

    # optimizer
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='bf16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
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
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler."
    )
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
    parser.add_argument(
        "--prodigy_decouple",
        action="store_true",
        help="Use AdamW style decoupled weight decay"
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-03,
        help="Weight decay to use for unet params"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        action="store_true",
        help="Turn on Adam's bias correction."
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )
    parser.add_argument("--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.")

    # misc
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    # validation
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )

    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


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


def save_model(model, global_step, args):
    """Save the unwrapped model."""
    if args.upcast_before_saving:
        model.to(torch.float32)

    if len(args.vpt_mode) > 0:
        vpt_path = os.path.join(args.output_dir, f"vpt-{global_step}.pt")
        torch.save(model.vpt_state_dict(), vpt_path)

    transformer_lora_layers = get_peft_model_state_dict(model)
    if args.train_norm_layers:
        transformer_norm_layers = {
            f"transformer.{name}": param
            for name, param in model.named_parameters()
            if any(k in name for k in NORM_LAYER_PREFIXES)
        }
        transformer_lora_layers = {**transformer_lora_layers, **transformer_norm_layers}

    HunyuanVideoPipeline.save_lora_weights(
        save_directory=args.output_dir,
        weight_name=f'pytorch_lora_weights_{global_step}.safetensors',
        transformer_lora_layers=transformer_lora_layers)


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str = None,
        prompt_embed_ann: str = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.scaling_factor = None # need to be filled later
        self.prompt_embed_ann = torch.load(prompt_embed_ann, weights_only=True)
        # find all video latents
        self.video_latent_files = sorted(glob.glob(os.path.join(root_dir, '*.pth')))
        self.video_latent_files = [v for v in self.video_latent_files if 'seed' in v]

    def __len__(self):
        return len(self.video_latent_files)

    def __getitem__(self, index):
        latent_path = self.video_latent_files[index]
        idx = latent_path[latent_path.rfind('/')+1:]
        idx = int(idx[:idx.find('_')])
        prompt_embed1, attention_mask1, prompt_embed2 = self.prompt_embed_ann[idx]
        video_latents = torch.load(latent_path, weights_only=True)
        video_latents = video_latents[0] * self.scaling_factor
        #video_latents = video_latents[:, :, :1] # only take the first frame
        first_latents = video_latents.clone()
        first_latents[:, :, 1:] = 0

        return {
            'prompt_embed1': prompt_embed1[0],
            'prompt_embed2': prompt_embed2[0],
            'attention_mask1': attention_mask1[0],
            'video_latents': video_latents,
            'image_latents': first_latents,
        }


@torch.no_grad
def visualize_latent(vae, z0):
    image = vae.decode(
        z0 / vae.config.scaling_factor,
        return_dict=False)[0]
    image = ((image[0].permute(1, 2, 3, 0) + 1) / 2 * 255).clamp(0, 255)
    return [Image.fromarray(x) for x in image.byte().cpu().numpy()]



def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    if args.use_lora_bias and args.gaussian_init_lora:
        raise ValueError("`gaussian` LoRA init scheme isn't supported when `use_lora_bias` is True.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)

    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=600.0))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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

    if accelerator.state.deepspeed_plugin:
        # Set deepspeed config according to args
        config = {
            'optimizer': {
                'type': args.optimizer,
                'params': {
                    'lr': args.learning_rate,
                    'betas': [args.adam_beta1, args.adam_beta2]
                },
                'torch_adam': True
            },
            'bf16': {
                'enabled': True if args.mixed_precision == "bf16" else False
            },
            'fp16': {
                'enabled': True if args.mixed_precision == "fp16" else False
            },
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'train_batch_size': args.train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes
        }
        accelerator.state.deepspeed_plugin.deepspeed_config.update(config)

    device = accelerator.device
    # cast down and move to the CPU
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
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
        rng = np.random.RandomState(args.seed)
        seed_ = rng.randint(1, 10000, size=(accelerator.num_processes,))
        set_seed(args.seed + seed_[accelerator.local_process_index])

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load models. We will load the text encoders later in a pipeline to compute
    # embeddings.
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype
    )
    transformer = MyHunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype
    )
    print(f"Set the mode of transformer rotary embedding to {args.rotary_mode} mode!")
    transformer.rope.set_mode(args.rotary_mode)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    # let's not move the VAE to the GPU yet.
    vae.to('cpu')
    transformer.to(dtype=weight_dtype, device=device)

    # enable image inputs
    """
    with torch.no_grad():
        initial_input_channels = transformer.config.in_channels
        new_linear = torch.nn.Linear(
            transformer.x_embedder.in_features * 2,
            transformer.x_embedder.out_features,
            bias=transformer.x_embedder.bias is not None,
            dtype=transformer.dtype,
            device=transformer.device,
        )
        new_linear.weight.zero_()
        new_linear.weight[:, :initial_input_channels].copy_(transformer.x_embedder.weight)
        if transformer.x_embedder.bias is not None:
            new_linear.bias.copy_(transformer.x_embedder.bias)
        transformer.x_embedder = new_linear

    assert torch.all(transformer.x_embedder.weight[:, initial_input_channels:].data == 0)
    transformer.register_to_config(in_channels=initial_input_channels * 2, out_channels=initial_input_channels)
    """

    if args.train_norm_layers:
        for name, param in transformer.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad = True

    if args.lora_layers is not None:
        if args.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
            # add the input layer to the mix.
            #if "x_embedder" not in target_modules:
            #    target_modules.append("x_embedder")
        elif args.lora_layers == "all-linear":
            target_modules = set()
            for name, module in transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=target_modules,
        lora_bias=args.use_lora_bias,
    )
    transformer.add_adapter(transformer_lora_config)

    if args.pretrained_lora_path:
        print(f'[ Loading pretrained LoRA weights from {args.pretrained_lora_path} ]')
        pipe = HunyuanVideoPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=weight_dtype).to(device)
        pipe.load_lora_weights(args.pretrained_lora_path, adapter_name="adapter")
        del pipe

    if len(args.vpt_mode) > 0:
        transformer.add_vpt(args.vpt_mode)
        if args.pretrained_path:
            print(f"Load vpt from {args.pretrained_path}")
            vpt_dic = torch.load(args.pretrained_path, weights_only=True, map_location='cpu')
            transformer.load_vpt(vpt_dic)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    if args.train_norm_layers:
                        transformer_norm_layers_to_save = {
                            f"transformer.{name}": param
                            for name, param in model.named_parameters()
                            if any(k in name for k in NORM_LAYER_PREFIXES)
                        }
                        transformer_lora_layers_to_save = {
                            **transformer_lora_layers_to_save,
                            **transformer_norm_layers_to_save,
                        }
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            HunyuanVideoPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

        else:
            transformer_ = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            ).to(device, weight_dtype)
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = HunyuanVideoPipeline.lora_state_dict(input_dir)
        transformer_lora_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.") and "lora" in k
        }
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_lora_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_norm_layers:
            transformer_norm_state_dict = {
                k: v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
            }
            transformer_._transformer_norm_layers = HunyuanVideoPipeline._load_norm_into_transformer(
                transformer_norm_state_dict,
                transformer=transformer_,
                discard_original_layers=False,
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    print(f"Use deep speed optimizer: {use_deepspeed_optimizer}")
    print(f"Use deep speed scheduler: {use_deepspeed_scheduler}")

    # Optimization parameters

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    params_to_optimize = []
    # use the same learning rate for all parameters
    if len(args.vpt_mode) > 0:
        transformer_lora_parameters.remove(transformer.vpt)
        transformer_lora_parameters.remove(transformer.vpt_scale)

        params_to_optimize.extend([{
                'params': [transformer.vpt],
                'lr': 1e-3
            } , {
                'params': [transformer.vpt_scale],
                'lr': 1e-3
            }])

    params_to_optimize.append({
            "params": transformer_lora_parameters,
            "lr": args.learning_rate
        })
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Prepare dataset and dataloader.
    train_dataset = VideoDataset(
        root_dir=args.data_dir,
        prompt_embed_ann=args.prompt_embed_ann
    )
    train_dataset.scaling_factor = vae.config.scaling_factor
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        drop_last=True,
        num_workers=0,
    )

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

    #from IPython import embed; embed()

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    x_timesteps, y_gradmaxs, z_losses = [], [], []
    image_logs = None
    while True:
        transformer.train()
        train_dataloader.reset()
        print("start new dataloading")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                latents = batch["video_latents"].to(dtype=weight_dtype).to(device)
                prompt_embed1 = batch["prompt_embed1"].to(dtype=weight_dtype).to(device)
                prompt_embed2 = batch["prompt_embed2"].to(dtype=weight_dtype).to(device)
                attention_mask1 = batch["attention_mask1"].to(device)
                image_latents = batch["image_latents"].to(device).to(dtype=weight_dtype)

                # text encoding: we will use pre-computed embedding to avoid loading text encoder
                """
                captions = batch["captions"]
                text_encoding_pipeline = text_encoding_pipeline.to("cuda")
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                        captions, prompt_2=None
                    )
                # this could be optimized by not having to do any text encoding and just
                # doing zeros on specified shapes for `prompt_embeds` and `pooled_prompt_embeds`
                if args.proportion_empty_prompts and random.random() < args.proportion_empty_prompts:
                    prompt_embeds.zero_()
                    pooled_prompt_embeds.zero_()
                """

                # Controlnet
                # control_latents = encode_images(batch["conditioning_pixel_values"], vae.to(device), weight_dtype)

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                batch_size = latents.shape[0]
                noise = torch.randn_like(latents, device=device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                t = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(t, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

                # controlnet training
                noisy_model_input = torch.cat([image_latents[:, :, :1], noisy_model_input], dim=2)

                guidance_vec = torch.tensor(
                    [args.embed_cfg_scale] * latents.shape[0],
                    dtype=weight_dtype, device=device) * 1000.0
                #guidance_vec = None

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embed1,
                    encoder_attention_mask=attention_mask1,
                    pooled_projections=prompt_embed2,
                    guidance=guidance_vec,
                    gc_ratio=args.gc_ratio,
                    return_dict=False)[0]
                model_pred = model_pred[:, :, 1:]

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow-matching loss
                target = noise - latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                with torch.no_grad():
                    grad_norm = [p.grad.norm() for p in transformer.parameters() if p.grad is not None]
                    grad_norm = float(sum(grad_norm) / len(grad_norm)) if grad_norm else 0.0

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or global_step == 1:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        """ # do not save the accelerator state because there is no use
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        """

                        save_model(unwrap_model(transformer), global_step, args)


            if global_step % 10 == 1:
                vae.enable_tiling()
                transformer.to('cpu'); vae.to(device)
                z0 = noisy_model_input[:, :, 1:] - model_pred * sigmas
                images = visualize_latent(vae, z0.bfloat16())
                export_to_video(images, f'{args.output_dir}/trainviz_{global_step}_{int(t[0])}.mp4', fps=30)
                vae.to('cpu'); transformer.to(device)

            if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                image_logs = log_validation(
                    transformer=transformer,
                    args=args,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    step=global_step,
                )

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
                "vpt_scale": float(transformer.module.vpt_scale.mean()),
                "vpt_norm": float(transformer.module.vpt.abs().mean()),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model(transformer, global_step, args)

        del transformer
        free_memory()

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                transformer=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*", "*.pt", "*.bin"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
