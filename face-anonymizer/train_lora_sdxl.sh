#!/usr/bin/env bash
set -euo pipefail

TRAIN_DATA_DIR="${TRAIN_DATA_DIR:?Set TRAIN_DATA_DIR to the imagefolder dataset directory.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the LoRA output.}"
DIFFUSERS_REPO="${DIFFUSERS_REPO:-$HOME/src/diffusers}"
BASE_MODEL="${BASE_MODEL:-SG161222/RealVisXL_V4.0}"
VALIDATION_PROMPT="${VALIDATION_PROMPT:-a photo of sks synthetic non-famous Korean adult face, natural skin texture}"
RESOLUTION="${RESOLUTION:-1024}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1200}"
RANK="${RANK:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"

TRAIN_SCRIPT="$DIFFUSERS_REPO/examples/text_to_image/train_text_to_image_lora_sdxl.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Could not find $TRAIN_SCRIPT. Clone https://github.com/huggingface/diffusers to $DIFFUSERS_REPO first." >&2
  exit 1
fi

accelerate launch "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path "$BASE_MODEL" \
  --train_data_dir "$TRAIN_DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --resolution "$RESOLUTION" \
  --center_crop \
  --random_flip \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --mixed_precision fp16 \
  --rank "$RANK" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --validation_prompt "$VALIDATION_PROMPT" \
  --num_validation_images 4 \
  --validation_epochs 1 \
  --checkpointing_steps 300 \
  --checkpoints_total_limit 3 \
  --seed 1234
