#!/bin/bash

MODE="bottleneck"
TRAIN_START_STEP=800
STOCHASTIC_LAYER=8
COND_TIMESTEP=800
LORA=16

EXPR_DIR="expr/test"

# if you are not using wth 8 gpus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
#accelerate launch --config_file finetune/accelerate_config_machine_single.yaml --multi_gpu finetune/train_i2v_controlnet_lora.py \
#ipython -i finetune/train_i2v_controlnet_lora.py -- \
#python -m pdb finetune/train_i2v_controlnet_lora.py \

#--pretrained_controlnet_path $CKPT_PATH/checkpoint-2000.pt --pretrained_lora_path $CKPT_PATH/pytorch_lora_weights_2000.safetensors \

CKPT_PATH=expr/cogvideox-i2v-5b/panohead-flame-controlnet-lora16-bottleneck-s800-sl8-ct800

#accelerate launch --config_file finetune/accelerate_config_machine_single.yaml --multi_gpu finetune/train_i2v_controlnet_lora.py \
python finetune/train_i2v.py \
  --gc_ratio 0.75 \
  --lora_rank $LORA --adapter lora \
  --ann_file $DATA_DIR/ann.parquet \
  --seed 42 \
  --data_root data/hunyuan_distillation_embedcfg6 \
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
