#!/bin/bash

#python finetune/train_i2v_diffusers.py \

#accelerate launch --config_file finetune/accelerate_config.yaml finetune/train_i2v_diffusers.py \

VPT="deep-add-1"
OUTPUT_DIR="expr/vpt-${VPT}"

# --pretrained_lora_path $LORA_PATH \


accelerate launch --config_file finetune/accelerate_config.yaml finetune/train_i2v_diffusers_vpt.py \
  --pretrained_model_name_or_path "hunyuanvideo-community/HunyuanVideo" \
  --output_dir $OUTPUT_DIR \
  --data_dir data/hunyuan_distillation_cfg6 \
  --prompt_embed_ann data/hunyuan_distillation_cfg6/new_prompts.pth \
  --cfg_scale 1.0 --embed_cfg_scale 6.0 \
  --resolution 720 \
  --mixed_precision "bf16" \
  --train_batch_size 1 \
  --vpt_mode $VPT \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing --gc_ratio 0.8 \
  --use_8bit_adam \
  --learning_rate 1e-3 \
  --lr_scheduler "constant" \
  --checkpointing_steps 50 \
  --lr_warmup_steps 50 \
  --max_train_steps 1000 \
  --seed 1000
