"""
Push  Multi-Task Model to HuggingFace Hub

Uploads the trained intent+urgency classifier checkpoint to HuggingFace.
Uses folder upload — preserves all jenga_ai files needed for inference.

Usage:
    python push_multitask_to_hub.py --token hf_xxxxxxxxxxxx

What gets pushed:
    checkpoints/best/   — best checkpoint (use this for inference)
    experiment_config.yaml
    model files (pytorch_model.bin or model.safetensors)
    tokenizer files
"""

import argparse
import os
from huggingface_hub import HfApi, login

MODEL_DIR  = "/root/sdk_training_runs/results/cpims-nlp-afribert-kenya-adapted" # runpod model path
BEST_CKPT  = os.path.join(MODEL_DIR, "checkpoints/best")
REPO_ID    = "Rogendo/cpims-nlp-intent-urgency"
PRIVATE    = True

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True, help="HuggingFace API token")
parser.add_argument("--best-only", action="store_true",
                    help="Push only the best checkpoint (smaller upload)")
args = parser.parse_args()

print("Logging in to HuggingFace...")
login(token=args.token)

api = HfApi()

print(f"Creating repo: {REPO_ID} (private={PRIVATE})")
api.create_repo(REPO_ID, private=PRIVATE, exist_ok=True, repo_type="model")

# Write a model card
model_card = """---
language:
  - sw
  - en
tags:
  - text-classification
  - multi-task
  - intent-classification
  - cpims
  - kenya
  - swahili
  - jenga-ai
base_model: Rogendo/afribert-kenya-adapted
---

# CPIMS Intent + Urgency Classifier

Multi-task NLP model for the Child Protection Information Management System (CPIMS) Kenya.

## Model Details

- **Base model**: [Rogendo/afribert-kenya-adapted](https://huggingface.co/Rogendo/afribert-kenya-adapted) (AfriBERT domain-adapted on Kenyan Swahili + Sheng + CPIMS corpus)
- **Architecture**: Multi-task classification — shared XLM-RoBERTa encoder with two classification heads
- **Task 1**: Intent classification (63 classes — CPIMS support intents)
- **Task 2**: Urgency classification (3 classes — high / medium / low)
- **Training data**: 1,465 CPIMS support messages (English, Swahili, code-switched Kenyan)
- **Framework**: [jenga_ai SDK](https://github.com/TujengeAI/JengaAI)


## Training Pipeline

1. Domain-Adaptive Pre-Training (MLM) on Swahili Wikipedia + MasakhaNEWS + CPIMS WhatsApp chats → `Rogendo/afribert-kenya-adapted`
2. Multi-task fine-tuning on CPIMS support messages → this model
"""

card_path = "/tmp/README.md"
with open(card_path, "w") as f:
    f.write(model_card)
api.upload_file(
    path_or_fileobj=card_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)
print("Model card uploaded.")

# Upload best checkpoint
upload_dir = BEST_CKPT if args.best_only else MODEL_DIR
label      = "best checkpoint only" if args.best_only else "full model directory"
print(f"\nUploading {label} from:\n  {upload_dir}")

api.upload_folder(
    folder_path=upload_dir,
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"\nDone! Model available at: https://huggingface.co/{REPO_ID}")
print(f"\nLoad in jenga_ai:")
print(f"  pipeline = InferencePipeline.from_checkpoint(")
print(f"      model_dir='{REPO_ID}',")
print(f"      config_path='experiment_config.yaml',")
print(f"  )")
