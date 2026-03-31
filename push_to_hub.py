"""
Push afribert-kenya-adapted to HuggingFace Hub.

Usage:
    python push_to_hub.py --token hf_xxxxxxxxxxxxxxxxxxxx
"""

import argparse
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import login

MODEL_DIR = "/workspace/DomainAdaptation/afribert-kenya-adapted/final"
REPO_ID   = "Rogendo/afribert-kenya-adapted"
PRIVATE   = True   # set False to make public

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True, help="HuggingFace API token (hf_xxx...)")
args = parser.parse_args()

print(f"Logging in to HuggingFace...")
login(token=args.token)

print(f"Loading model from {MODEL_DIR}...")
model     = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f"Pushing to {REPO_ID} (private={PRIVATE})...")
model.push_to_hub(REPO_ID, private=PRIVATE)
tokenizer.push_to_hub(REPO_ID, private=PRIVATE)

print(f"\nDone! Model available at: https://huggingface.co/{REPO_ID}")
print(f"\nUse in jenga_ai YAML:")
print(f"  model:")
print(f"    base_model: {REPO_ID}")
