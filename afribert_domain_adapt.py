"""
AfriBERT Kenya Domain Adaptation via Masked Language Modeling

Adapts castorini/afriberta_large to the Kenyan language domain using
five complementary data sources:

  Source                    Type              Est. tokens    Why it matters
  ─────────────
  1. Swahili Wikipedia      Encyclopedic      ~22M           Factual Swahili prose
  2. CC-100 Swahili         Web crawl         ~15M           Kenyan web text, news, blogs
  3. MasakhaNEWS Swahili    News articles     ~1M            Formal East African journalism
  4. WhatsApp CPIMS chat    CPIMS field data  ~30K           Real field worker language
  5. Synthetic corpus       Code-switch/Sheng ~1M (×10)      M-PESA, CPIMS, Sheng patterns

  TOTAL                                       ~39M tokens

Sources 4 and 5 are repeated at higher frequency so the model sees
domain-specific Kenyan language proportionally more than generic Wikipedia.

Output: standard HuggingFace model directory → use as base_model in jenga_ai.

RunPod A40 estimated time: ~40-55 minutes total (incl. all downloads)
RunPod A40 estimated cost: ~$0.55-0.70 at $0.79/hr

Usage (RunPod):
    # 1. Start A40 pod (PyTorch 2.x template)
    # 2. pip install -r requirements_mlm.txt
    # 3. Upload master_mlm_corpus.txt    → /workspace/
    # 4. Upload cpims_support_whatsappchat.txt → /workspace/
    # 5. python afribert_domain_adapt.py

Requirements:
    torch>=2.1.0
    transformers>=4.40.0
    datasets>=2.18.0
    accelerate>=0.27.0
    sentencepiece>=0.1.99
"""

import os
import re
import time
import math
import itertools
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# CONFIGURATION — edit paths and toggles before running


BASE_MODEL = "castorini/afriberta_large"

# ── Local files (upload to /workspace/ on RunPod) 
SYNTHETIC_CORPUS_PATH = "/workspace/DomainAdaptation/master_mlm_corpus.txt"
WHATSAPP_CORPUS_PATH  = "/workspace/DomainAdaptation/cpims_support_whatsappchat.txt"

# ── Corpus toggles (set False to skip a source) 
USE_WIKIPEDIA   = True
USE_CC100       = False   # gated dataset — requires HF auth    # CC-100 Swahili — streams first CC100_LIMIT lines
USE_MASAKHANEWS = True    # MasakhaNEWS East African news (~5K articles)
USE_WHATSAPP    = True    # CPIMS WhatsApp chat (requires file above)

# ── Repetition — how many times to repeat each domain-specific source ─────────
# Higher = model sees that source proportionally more vs Wikipedia/CC-100
SYNTHETIC_REPEAT = 10   # Sheng/code-switch synthetic data
WHATSAPP_REPEAT  = 20   # Real CPIMS field worker messages (very small → boost heavily)
NEWS_REPEAT      = 3    # MasakhaNEWS — formal Swahili journalism

# ── CC-100 stream limit ──
# 100_000 lines ≈ 15M tokens, ~5 min stream on A40
# Set to None to stream the full dataset (much slower, not recommended)
CC100_LIMIT = 100_000

# ── Training ──────
BLOCK_SIZE    = 128   # Match downstream classification max_length
BATCH_SIZE    = 64    # A40 48GB handles this comfortably at seq_len=128 + bf16
GRAD_ACCUM    = 1
NUM_EPOCHS    = 3
LEARNING_RATE = 1e-4  # Higher than fine-tuning LR — standard for continued pre-training
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.06  # 6% of total steps
MLM_PROB      = 0.15  # Standard BERT masking: 15% of tokens masked per sample

# ── Evaluation ────
EVAL_SPLIT      = 0.05   # 5% held out for validation
EVAL_BATCH_SIZE = 128

# ── Output ────────
OUTPUT_DIR    = "/workspace/DomainAdaptation/afribert-kenya-adapted"
LOGGING_STEPS = 200




# ══ Loaders ════

def load_synthetic(path: str, repeat: int) -> list[str]:
    """Local synthetic Sheng / Swahili-English code-switch corpus."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nSynthetic corpus not found: {path}\n"
            "Upload master_mlm_corpus.txt to /workspace/ on RunPod.\n"
        )
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if len(l.strip()) > 20]
    log.info(f"[synthetic]     {len(lines):>7,} lines  →  ×{repeat} = {len(lines)*repeat:,} texts")
    return lines * repeat


def load_whatsapp(path: str, repeat: int) -> list[str]:
    """
    Parse WhatsApp export — strip timestamps/sender names, keep messages only.
    Expected format:  dd/mm/yyyy, HH:MM - Sender Name: message text
    """
    if not os.path.exists(path):
        log.warning(f"[whatsapp]      File not found: {path} — skipping")
        return []

    
    header = re.compile(r"^\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s[aApP][mM]\s-\s[^:]+:\s")
    messages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if header.match(line):
                msg = header.sub("", line).strip()
                if len(msg) > 20 and "<Media omitted>" not in msg:
                    messages.append(msg)

    if not messages:
        log.warning("[whatsapp]      No messages parsed — check file format")
        return []

    log.info(f"[whatsapp]      {len(messages):>7,} messages  →  ×{repeat} = {len(messages)*repeat:,} texts")
    return messages * repeat


def load_wikipedia(config: str) -> list[str]:
    """Full Swahili Wikipedia — all articles longer than 200 characters."""
    from datasets import load_dataset

    log.info(f"[wikipedia]     Downloading {config} — 5-15 min on first run...")
    t0 = time.time()
    # FIX: trust_remote_code not needed for wikimedia/wikipedia
    wiki = load_dataset("wikimedia/wikipedia", config, split="train")
    texts = [item["text"] for item in wiki if len(item["text"]) > 200]
    log.info(f"[wikipedia]     {len(texts):>7,} articles kept  ({time.time()-t0:.0f}s)")
    return texts


def load_cc100(limit: int) -> list[str]:
    """
    CC-100 Swahili — streamed to avoid downloading the full ~900MB corpus.
    Streams the first `limit` paragraphs only.

    FIX: "sw" is a positional config arg, NOT lang= keyword.
    FIX: Dataset ID is "statmt/cc100" (canonical) or "cc100" (alias).
    """
    from datasets import load_dataset

    log.info(f"[culturax]      Streaming {limit:,} lines from CulturaX/sw — 3-8 min...")
    t0 = time.time()

    # FIX: positional config "sw", not lang="sw"
    ds = load_dataset("uonlp/CulturaX", "sw", split="train", streaming=True)

    texts = []
    for item in itertools.islice(ds, limit):
        text = item["text"].strip()
        if len(text) > 30:
            texts.append(text)

    log.info(f"[culturax]      {len(texts):>7,} lines kept  ({time.time()-t0:.0f}s)")
    return texts


def load_masakhanews(repeat: int) -> list[str]:
    """
    MasakhaNEWS Swahili — all splits concatenated.
    ~5K East African news articles in formal Swahili.

    FIX: config is "swa" (ISO 639-3), not "sw".
    Columns: category, headline, text, url
    """
    from datasets import load_dataset, concatenate_datasets

    log.info("[masakhanews]   Loading East African news corpus...")
    t0 = time.time()

    splits = []
    # FIX: config is "swa", not "sw"
    for split_name in ("train", "validation", "test"):
        try:
            ds = load_dataset("masakhane/masakhanews", "swa", split=split_name)
            splits.append(ds)
        except Exception as e:
            log.warning(f"[masakhanews]   Could not load split '{split_name}': {e}")

    if not splits:
        log.warning("[masakhanews]   All splits failed — skipping")
        return []

    combined = concatenate_datasets(splits)

    # Combine headline + body for richer context per article
    texts = []
    for item in combined:
        headline = (item.get("headline") or "").strip()
        body     = (item.get("text") or "").strip()
        full     = f"{headline}. {body}" if headline and body else (headline or body)
        if len(full) > 30:
            texts.append(full)

    log.info(
        f"[masakhanews]   {len(texts):>7,} articles  →  ×{repeat} = {len(texts)*repeat:,} texts"
        f"  ({time.time()-t0:.0f}s)"
    )
    return texts * repeat


# ══ Dataset builder ════

def build_dataset(all_texts: list[str], tokenizer):
    """Tokenize all texts then chunk into fixed BLOCK_SIZE sequences."""
    from datasets import Dataset

    log.info(f"Building MLM dataset from {len(all_texts):,} texts...")
    t0 = time.time()

    raw = Dataset.from_dict({"text": all_texts})

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=False,   # No truncation — we chunk manually below
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    tokenized = raw.map(
        tokenize,
        batched=True,
        batch_size=2000,
        num_proc=1,             # Keep at 1 to avoid multiprocessing deadlocks
        remove_columns=["text"],
        desc="Tokenizing",
    )
    log.info(f"Tokenized {len(tokenized):,} sequences in {time.time()-t0:.0f}s")

    def chunk(examples):
        # FIX: use itertools.chain (O(n)) instead of sum([], []) which is O(n²)
        all_ids = list(itertools.chain.from_iterable(examples["input_ids"]))
        total   = (len(all_ids) // BLOCK_SIZE) * BLOCK_SIZE  # drop small remainder
        chunks  = [all_ids[i: i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
        # All-1 attention mask — no padding in fixed-size chunks
        masks   = [[1] * BLOCK_SIZE for _ in chunks]
        return {"input_ids": chunks, "attention_mask": masks}

    chunked = tokenized.map(
        chunk,
        batched=True,
        batch_size=1000,
        num_proc=1,
        desc="Chunking",
    )
    log.info(
        f"Dataset ready: {len(chunked):,} chunks × {BLOCK_SIZE} tokens"
        f" = {len(chunked) * BLOCK_SIZE:,} total tokens"
    )
    return chunked


# ══ Estimator ═══

def print_estimate(n_chunks: int) -> None:
    steps_per_epoch = math.ceil(n_chunks / BATCH_SIZE)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    total_secs      = total_steps * 0.065   # ~65ms/step on A40, bf16, seq_len=128

    log.info("─" * 62)
    log.info("TRAINING ESTIMATE  (A40 GPU, bf16, batch=%d, block=%d)", BATCH_SIZE, BLOCK_SIZE)
    log.info("  Train chunks    : %s", f"{n_chunks:,}")
    log.info("  Steps / epoch   : %s", f"{steps_per_epoch:,}")
    log.info("  Total steps     : %s", f"{total_steps:,}")
    log.info("  Est. train time : %.1f min", total_secs / 60)
    log.info("  RunPod A40 cost : ~$%.2f  (at $0.79/hr)", total_secs / 3600 * 0.79)
    log.info("─" * 62)


# ══ Main ═══

def main():
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )

    wall_start = time.time()
    log.info("=" * 62)
    log.info("AfriBERT Kenya Domain Adaptation — MLM Pre-training")
    log.info("  Base model : %s", BASE_MODEL)
    log.info("  Block size : %d", BLOCK_SIZE)
    log.info("  Epochs     : %d", NUM_EPOCHS)
    log.info("  Output     : %s", OUTPUT_DIR)
    log.info("=" * 62)

    # ── 1. Load tokenizer + model ──
    log.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model loaded: {n_params:.1f}M parameters")

    # ── 2. Assemble corpus ─
    log.info("")
    log.info("── Loading corpora ──")
    all_texts = []

    # Source 1: Synthetic Sheng/code-switch (domain-specific, repeated)
    all_texts.extend(load_synthetic(SYNTHETIC_CORPUS_PATH, SYNTHETIC_REPEAT))

    # Source 2: CPIMS WhatsApp chat (tiny but domain-gold, repeated heavily)
    if USE_WHATSAPP:
        all_texts.extend(load_whatsapp(WHATSAPP_CORPUS_PATH, WHATSAPP_REPEAT))

    # Source 3: MasakhaNEWS — formal East African Swahili journalism
    if USE_MASAKHANEWS:
        all_texts.extend(load_masakhanews(NEWS_REPEAT))

    # Source 4: CC-100 Swahili — large Kenyan web text
    if USE_CC100:
        all_texts.extend(load_cc100(CC100_LIMIT))

    # Source 5: Swahili Wikipedia — full encyclopedic Swahili
    if USE_WIKIPEDIA:
        all_texts.extend(load_wikipedia("20231101.sw"))

    log.info("─" * 62)
    log.info(f"TOTAL corpus : {len(all_texts):,} texts")
    log.info("")

    # ── 3. Tokenize + chunk ──
    full_dataset = build_dataset(all_texts, tokenizer)

    split         = full_dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    log.info(f"Train: {len(train_dataset):,} chunks | Eval: {len(eval_dataset):,} chunks")

    # ── 4. Time/cost estimate ──
    print_estimate(len(train_dataset))

    # ── 5. Data collator — dynamic masking (15% tokens per batch) ────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    # ── 6. Compute total steps (needed for warmup_steps below) ────
    steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    total_steps     = steps_per_epoch * NUM_EPOCHS

    # ── 6b. Device detection ───
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name}  ({vram_gb:.0f} GB VRAM)")
        use_bf16 = True
        use_fp16 = False
    else:
        log.warning("No GPU detected — training will be very slow on CPU")
        use_bf16 = False
        use_fp16 = False

    # ── 7. Training arguments ────
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = EVAL_BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        warmup_steps                = int(WARMUP_RATIO * total_steps),
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        eval_strategy               = "epoch",   # transformers>=4.40 uses eval_strategy
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_steps               = LOGGING_STEPS,
        report_to                   = "none",
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
        seed                        = 42,
    )

    # ── 8. Trainer 
    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        data_collator = data_collator,
        processing_class = tokenizer,
    )

    # ── 9. Train ──
    log.info("Starting MLM training...")
    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start
    log.info(f"Training complete in {train_elapsed/60:.1f} minutes")

    # ── 10. Evaluate + perplexity ───
    log.info("Evaluating on held-out set...")
    eval_results = trainer.evaluate()
    perplexity   = math.exp(eval_results["eval_loss"])
    log.info(f"Eval loss  : {eval_results['eval_loss']:.4f}")
    log.info(f"Perplexity : {perplexity:.2f}")
    log.info("  Baseline (AfriBERT, no DAPT): ~80-200 on Kenyan domain text")
    log.info("  Target after DAPT           : ~20-60")

    # ── 11. Save adapted model ─────
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    log.info(f"Saving adapted model → {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Write a corpus provenance file alongside the model
    with open(os.path.join(final_dir, "corpus_summary.txt"), "w") as f:
        f.write("AfriBERT Kenya DAPT — Corpus Summary\n")
        f.write("=" * 42 + "\n")
        f.write(f"Base model       : {BASE_MODEL}\n")
        f.write(f"Block size       : {BLOCK_SIZE}\n")
        f.write(f"Epochs           : {NUM_EPOCHS}\n")
        f.write(f"Total texts      : {len(all_texts):,}\n")
        f.write(f"Train chunks     : {len(train_dataset):,}\n")
        f.write(f"Final perplexity : {perplexity:.2f}\n")
        f.write(f"Training time    : {train_elapsed/60:.1f} min\n\n")
        f.write("Sources:\n")
        f.write(f"  [1] Wikipedia sw 20231101          : {'YES' if USE_WIKIPEDIA else 'NO'}\n")
        f.write(f"  [2] CC-100 sw ({CC100_LIMIT:,} lines)      : {'YES' if USE_CC100 else 'NO'}\n")
        f.write(f"  [3] MasakhaNEWS swa  (×{NEWS_REPEAT})        : {'YES' if USE_MASAKHANEWS else 'NO'}\n")
        f.write(f"  [4] WhatsApp CPIMS chat (×{WHATSAPP_REPEAT})   : {'YES' if USE_WHATSAPP else 'NO'}\n")
        f.write(f"  [5] Synthetic Sheng corpus (×{SYNTHETIC_REPEAT})  : YES\n")

    # ── 12. Final summary ─────
    wall_elapsed = time.time() - wall_start
    log.info("=" * 62)
    log.info("DONE")
    log.info(f"  Total wall time  : {wall_elapsed/60:.1f} min")
    log.info(f"  Training time    : {train_elapsed/60:.1f} min")
    log.info(f"  Final perplexity : {perplexity:.2f}")
    log.info(f"  Adapted model    : {final_dir}")
    log.info("")
    log.info("Next step — update jenga_ai YAML:")
    log.info("  model:")
    log.info(f"    base_model: {final_dir}")
    log.info("=" * 62)


if __name__ == "__main__":
    main()
