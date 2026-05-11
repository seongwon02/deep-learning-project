# Synthetic Identity LoRA Training

Use this only after the base anonymization pipeline is running. The strongest demo comes from a small bank of synthetic identities, not one universal LoRA.

## Data

For each synthetic identity, prepare 20-80 licensed or generated training images with varied pose, lighting, focal length, expression, and crop.

Recommended imagefolder layout:

```text
data/synthetic_male_01/
  metadata.jsonl
  0001.png
  0002.png
```

Example `metadata.jsonl`:

```jsonl
{"file_name":"0001.png","text":"a photo of sks synthetic_male_01, synthetic non-famous Korean man face, natural skin texture"}
{"file_name":"0002.png","text":"a candid photo of sks synthetic_male_01, synthetic non-famous Korean man face"}
```

Use a unique token per identity, such as `sks synthetic_male_01`.

## Windows Training

Clone Diffusers examples once:

```powershell
git clone https://github.com/huggingface/diffusers $env:USERPROFILE\src\diffusers
```

Run:

```powershell
.\tools\face_anonymizer\train_lora_sdxl.ps1 `
  -TrainDataDir data\synthetic_male_01 `
  -OutputDir local\loras\synthetic_male_01 `
  -ValidationPrompt "a photo of sks synthetic_male_01, synthetic non-famous Korean man face, natural skin texture"
```

Then point `identity_bank.example.json` at the LoRA output directory.

## Practical Settings

Start with:

```text
rank: 16
steps: 800-1500
lora_weight at inference: 0.65-0.85
base model for training: SG161222/RealVisXL_V4.0
inpaint model for inference: OzzyGT/RealVisXL_V4.0_inpainting
```

Train multiple identities and assign them per `track_id`. If one LoRA overfits, lower `lora_weight`, reduce steps, or increase dataset variation.
