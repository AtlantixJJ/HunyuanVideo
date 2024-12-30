#!/bin/bash


LORA=64


#python finetune/train_i2v.py \
accelerate launch finetune/train_i2v.py \
  --lora_rank $LORA --adapter lora \
  --seed 42 \
  --video-width 720 --video-height 720 \
  --data_root data/hunyuan_distillation_cfg1 \
  --output_dir $EXPR_DIR \
  --train_start_step $TRAIN_START_STEP \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --max_train_steps 2000 \
  --checkpointing_steps 50 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --optimizer AdamW \
  --allow_tf32
