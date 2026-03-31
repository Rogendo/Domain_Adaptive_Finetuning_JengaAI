"""
AfriBERT Domain Adaptation — Before vs After Comparison
========================================================
Compares:
  - Base model : castorini/afriberta_large  (no DAPT)
  - Adapted    : Rogendo/afribert-kenya-adapted  (Kenya DAPT)

Tests:
  1. Masked token prediction — top-5 predictions per sentence
  2. Perplexity comparison — lower = model understands the text better

Usage:
    python compare_models.py
"""

import math
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

BASE_MODEL    = "castorini/afriberta_large"
ADAPTED_MODEL = "Rogendo/afribert-kenya-adapted"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Test sentences — one word replaced with <mask>
# Covers: Swahili, Sheng, M-PESA, CPIMS, code-switching

MASK_TESTS = [
    {
        "label": "Standard Swahili — Wikipedia style",
        "sentence": "Akiolojia ni somo linalohusu mabaki ya tamaduni za watu wa nyakati zilizopita.",
        "mask_word": "tamaduni",
    },
    {
        "label": "East African news — formal",
        "sentence": "Serikali imetangaza mpango mpya wa kukuza uchumi wa taifa kupitia biashara ya kimataifa.",
        "mask_word": "uchumi",
    },
    {
        "label": "M-PESA domain — financial",
        "sentence": "Tuma pesa kwa kutumia nambari ya simu kupitia huduma ya M-PESA.",
        "mask_word": "pesa",
    },
    {
        "label": "CPIMS domain — child protection",
        "sentence": "Mtoto aliripotiwa kwa ofisi ya ustawi wa jamii baada ya kudhulumiwa nyumbani.",
        "mask_word": "ustawi",
    },
    {
        "label": "Sheng / code-switching — Nairobi urban",
        "sentence": "Msee alikuwa poa sana, akanisaidia kupata kazi ya ofisi.",
        "mask_word": "kazi",
    },
    {
        "label": "WhatsApp CPIMS — field worker message",
        "sentence": "Mtoto huyu ana umri wa miaka kumi na mbili na anahitaji msaada wa haraka.",
        "mask_word": "msaada",
    },
    {
        "label": "English-Swahili code-switch",
        "sentence": "Tulifanya meeting jana na manager akasema project itakuwa ready wiki ijayo.",
        "mask_word": "project",
    },
]


# Perplexity test sentences (no masking — full sentences)

PERPLEXITY_TESTS = [
    "Tuma pesa kwa kutumia nambari ya simu kupitia huduma ya M-PESA.",
    "Mtoto aliripotiwa kwa ofisi ya ustawi wa jamii baada ya kudhulumiwa nyumbani.",
    "Msee alikuwa poa sana, akanisaidia kupata kazi ya ofisi mjini Nairobi.",
    "Serikali imetangaza mpango mpya wa kukuza uchumi wa taifa kupitia biashara ya kimataifa.",
    "Akiolojia ni somo linalohusu mabaki ya tamaduni za watu wa nyakati zilizopita.",
    "Tulifanya meeting jana na manager akasema project itakuwa ready wiki ijayo.",
    "Watoto wengi wanakabiliwa na changamoto za elimu na afya katika maeneo ya vijijini.",
    "Ninahitaji kuweka akiba yangu salama kupitia akaunti ya benki au huduma za kifedha.",
]




def load_model(model_name: str):
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE).eval()
    return tokenizer, model


def predict_masked(sentence: str, mask_word: str, tokenizer, model, top_k: int = 5):
    masked = sentence.replace(mask_word, tokenizer.mask_token)
    inputs = tokenizer(masked, return_tensors="pt").to(DEVICE)

    mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    if mask_idx.numel() == 0:
        return []

    with torch.no_grad():
        logits = model(**inputs).logits

    probs    = torch.softmax(logits[0, mask_idx[0]], dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)
    return [(tokenizer.decode([i.item()]).strip(), p.item()) for i, p in zip(top_ids, top_probs)]


def compute_perplexity(text: str, tokenizer, model) -> float:
    """Compute pseudo-perplexity via sequential token masking (MLM perplexity)."""
    inputs   = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"][0]
    n_tokens  = input_ids.size(0) - 2   # exclude [CLS] and [SEP]

    total_log_prob = 0.0
    for i in range(1, n_tokens + 1):   # skip position 0 ([CLS])
        masked_ids = input_ids.clone()
        masked_ids[i] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked_ids.unsqueeze(0)).logits

        log_probs   = torch.log_softmax(logits[0, i], dim=-1)
        true_token  = input_ids[i].item()
        total_log_prob += log_probs[true_token].item()

    avg_neg_log_prob = -total_log_prob / n_tokens
    return math.exp(avg_neg_log_prob)


def run_mask_comparison(base_tok, base_model, ada_tok, ada_model):
    print("\n" + "=" * 70)
    print("MASKED TOKEN PREDICTION — BASE vs ADAPTED")
    print("=" * 70)

    for test in MASK_TESTS:
        print(f"\n[{test['label']}]")
        print(f"  Sentence  : {test['sentence']}")
        print(f"  Masked    : {test['mask_word']}")

        base_preds = predict_masked(test["sentence"], test["mask_word"], base_tok, base_model)
        ada_preds  = predict_masked(test["sentence"], test["mask_word"], ada_tok,  ada_model)

        print(f"\n  {'BASE MODEL':<35}  {'ADAPTED MODEL'}")
        print(f"  {'-'*33}  {'-'*33}")
        for i, ((bt, bp), (at, ap)) in enumerate(zip(base_preds, ada_preds)):
            print(f"  {i+1}. {bt:<20} ({bp:.3f})    {at:<20} ({ap:.3f})")


def run_perplexity_comparison(base_tok, base_model, ada_tok, ada_model):
    print("\n" + "=" * 70)
    print("PERPLEXITY COMPARISON — lower is better")
    print("  (pseudo-perplexity via sequential MLM masking)")
    print("=" * 70)
    print(f"\n  {'Sentence':<55} {'Base':>8}  {'Adapted':>8}  {'Δ':>8}")
    print(f"  {'-'*53}  {'-'*8}  {'-'*8}  {'-'*8}")

    base_total = 0.0
    ada_total  = 0.0

    for text in PERPLEXITY_TESTS:
        base_ppl = compute_perplexity(text, base_tok, base_model)
        ada_ppl  = compute_perplexity(text, ada_tok,  ada_model)
        delta    = ada_ppl - base_ppl
        short    = text[:52] + "..." if len(text) > 55 else text
        direction = "↓ better" if delta < 0 else "↑ worse"
        print(f"  {short:<55} {base_ppl:>8.1f}  {ada_ppl:>8.1f}  {delta:>+7.1f} {direction}")
        base_total += base_ppl
        ada_total  += ada_ppl

    n = len(PERPLEXITY_TESTS)
    print(f"\n  {'AVERAGE':<55} {base_total/n:>8.1f}  {ada_total/n:>8.1f}  {(ada_total-base_total)/n:>+7.1f}")
    improvement = (base_total - ada_total) / base_total * 100
    print(f"\n  Domain adaptation improved average perplexity by {improvement:.1f}%")


def main():
    print(f"Device: {DEVICE.upper()}")
    print(f"\nLoading models...")

    base_tok, base_model = load_model(BASE_MODEL)
    ada_tok,  ada_model  = load_model(ADAPTED_MODEL)

    run_mask_comparison(base_tok, base_model, ada_tok, ada_model)
    run_perplexity_comparison(base_tok, base_model, ada_tok, ada_model)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
