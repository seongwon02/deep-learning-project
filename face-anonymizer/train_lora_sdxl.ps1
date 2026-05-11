param(
  [Parameter(Mandatory = $true)]
  [string]$TrainDataDir,

  [Parameter(Mandatory = $true)]
  [string]$OutputDir,

  [string]$DiffusersRepo = "$env:USERPROFILE\src\diffusers",
  [string]$BaseModel = "SG161222/RealVisXL_V4.0",
  [string]$ValidationPrompt = "a photo of sks synthetic non-famous Korean adult face, natural skin texture",
  [int]$Resolution = 1024,
  [int]$MaxTrainSteps = 1200,
  [int]$Rank = 16,
  [double]$LearningRate = 0.0001
)

$TrainScript = Join-Path $DiffusersRepo "examples\text_to_image\train_text_to_image_lora_sdxl.py"
if (!(Test-Path $TrainScript)) {
  throw "Could not find $TrainScript. Clone https://github.com/huggingface/diffusers to $DiffusersRepo first."
}

accelerate launch $TrainScript `
  --pretrained_model_name_or_path $BaseModel `
  --train_data_dir $TrainDataDir `
  --output_dir $OutputDir `
  --resolution $Resolution `
  --center_crop `
  --random_flip `
  --train_batch_size 1 `
  --gradient_accumulation_steps 4 `
  --gradient_checkpointing `
  --mixed_precision fp16 `
  --rank $Rank `
  --learning_rate $LearningRate `
  --lr_scheduler cosine `
  --lr_warmup_steps 100 `
  --max_train_steps $MaxTrainSteps `
  --validation_prompt $ValidationPrompt `
  --num_validation_images 4 `
  --validation_epochs 1 `
  --checkpointing_steps 300 `
  --checkpoints_total_limit 3 `
  --seed 1234
