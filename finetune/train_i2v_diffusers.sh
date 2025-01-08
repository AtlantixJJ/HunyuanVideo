#!/bin/bash

#python finetune/train_i2v_diffusers.py \

#accelerate launch --config_file finetune/accelerate_config.yaml finetune/train_i2v_diffusers.py \

RANK=128
ROTARY="i2v-temporal-2"
VPT="deep-add-1"
PROJ_NAME="lora-${RANK}_vpt-${VPT}_rotary-${ROTARY}"
OUTPUT_DIR="expr/${PROJ_NAME}"
LORA_PATH="expr/${PROJ_NAME}/pytorch_lora_weights_50.safetensors"

# --pretrained_lora_path $LORA_PATH \


accelerate launch --config_file finetune/accelerate_config.yaml --main_process_port 25000 finetune/train_i2v_diffusers.py \
  --pretrained_model_name_or_path "hunyuanvideo-community/HunyuanVideo" \
  --output_dir ${OUTPUT_DIR} \
  --data_dir data/hunyuan_distillation_cfg6 \
  --prompt_embed_ann data/hunyuan_distillation_cfg6/new_prompts.pth \
  --tracker_project_name ${PROJ_NAME} \
  --cfg_scale 1.0 --embed_cfg_scale 6.0 \
  --resolution 720 \
  --mixed_precision "bf16" \
  --train_batch_size 1 \
  --rank ${RANK} --lora_layers all-linear \
  --vpt_mode ${VPT} \
  --rotary_mode ${ROTARY} \
  --logit_mean 2.0 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing --gc_ratio 0.9 \
  --use_8bit_adam \
  --learning_rate 1e-4 \
  --lr_scheduler "constant" \
  --checkpointing_steps 50 \
  --lr_warmup_steps 50 \
  --max_train_steps 1000 \
  --seed 1000
