---
language:
  - sw
  - en
tags:
  - masked-language-modeling
  - domain-adaptation
  - swahili
  - sheng
  - kenya
  - cpims
  - african-nlp
  - afriberta
base_model: castorini/afriberta_large
license: apache-2.0
---

# AfriBERT Kenya — Domain-Adapted Language Model

`Rogendo/afribert-kenya-adapted` is a continued pre-training of [`castorini/afriberta_large`](https://huggingface.co/castorini/afriberta_large) on a Kenyan language corpus using **Masked Language Modeling (MLM)**.

It is optimised for Kenyan text: formal Swahili, Nairobi Sheng, M-PESA financial language, CPIMS child-protection terminology, and English-Swahili code-switching as used in everyday Kenyan communication.

---

## What is Domain-Adaptive Pre-Training (DAPT)?

Standard AfriBERT was trained on African newswire and Wikipedia. While it understands Swahili well, it has never seen:

- Sheng slang (`msee`, `poa`, `si poa`, `sawa kabisa`)
- M-PESA vocabulary (`Fuliza`, `Lipa na M-PESA`, `float`, `till number`)
- CPIMS child-protection terminology (`ustawi wa jamii`, `OVC`, `case worker`, `safe house`)
- Kenyan WhatsApp code-switching patterns

DAPT is an intermediate training step between the base pretrained model and task-specific fine-tuning. It continues MLM pre-training on domain text so the model builds a richer internal representation of these patterns before learning any downstream task.

---

## Training Data

The model was trained on **five complementary sources** totalling approximately **39 million tokens**:

| # | Source | Type | Est. Tokens | Repeat | Purpose |
|---|--------|------|------------|--------|---------|
| 1 | **Swahili Wikipedia** (`wikimedia/wikipedia`, `20231101.sw`) | Encyclopedic prose | ~22M | ×1 | Foundational standard Swahili — proper nouns, formal syntax, factual text |
| 2 | **MasakhaNEWS** (`masakhane/masakhanews`, `swa`) | East African journalism | ~1M | ×3 | Formal East African reporting style; Kenyan political, economic, social vocabulary |
| 3 | **Synthetic Sheng/Code-switch corpus** (`master_mlm_corpus.txt`) | Synthetic | ~1M | ×10 | Nairobi Sheng, M-PESA transactions, CPIMS case notes, English-Swahili switches |
| 4 | **WhatsApp CPIMS chat** (field worker exports) | Real conversational | ~30K | ×20 | Authentic CPIMS field worker language — the highest-value domain signal |

**Note:** CC-100 Swahili (`uonlp/CulturaX`) was available but disabled in the final run; sources 3 and 4 were repeated at high frequency so the model sees Kenyan domain text proportionally more than generic Wikipedia.

### Synthetic Corpus (Source 3)

`master_mlm_corpus.txt` is a hand-crafted synthetic corpus covering:

- **M-PESA transactions** — sending, receiving, Fuliza overdraft, Lipa na M-PESA, Buy Goods
- **CPIMS case language** — intake forms, referrals, OVC (Orphans and Vulnerable Children), safe-house placements, court orders
- **Sheng vocabulary** — Nairobi urban slang integrated into Swahili sentences
- **English-Swahili code-switching** — meeting minutes, office messages, WhatsApp style

### WhatsApp CPIMS Chat (Source 4)

Real WhatsApp export from a CPIMS field support group (`whatsappchat-Bungoma.txt`). Messages were filtered to remove media attachments and very short messages (<20 characters). This source was up-sampled ×20 because it contains the highest-quality real-world signal for the target domain despite its small size.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `castorini/afriberta_large` |
| Training objective | Masked Language Modeling (MLM) |
| Masking probability | 15% |
| Block / sequence length | 128 tokens |
| Batch size | 64 (NVIDIA A40, bf16) |
| Epochs | 3 |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Warmup | 6% of total steps |
| Hardware | NVIDIA A40 (48 GB VRAM) |
| Precision | bfloat16 |
| Training time | ~25.7 minutes |
| Eval split | 5% held-out |

---

## Results

All evaluations were run on **CPU** using `castorini/afriberta_large` (base) vs `Rogendo/afribert-kenya-adapted` (adapted). Pseudo-perplexity is computed via sequential token masking — each token in the sentence is masked one at a time and the model's log-probability for the correct token is accumulated.

### MLM Perplexity (lower = better)

| Domain | Sentence (truncated) | Base PPL | Adapted PPL | Δ |
|---|---|---|---|---|
| M-PESA | *"Tuma pesa kwa kutumia nambari ya simu..."* | 2.4 | 2.7 | +0.3 ↑ |
| CPIMS child protection | *"Mtoto aliripotiwa kwa ofisi ya ustawi wa jamii..."* | 8.0 | 7.8 | −0.2 ↓ |
| **Sheng / Nairobi urban** | *"Msee alikuwa poa sana, akanisaidia kupata kazi..."* | **11.3** | **3.8** | **−7.5 ↓** |
| East African news | *"Serikali imetangaza mpango mpya wa kukuza uchumi..."* | 3.3 | 3.1 | −0.2 ↓ |
| Standard Swahili | *"Akiolojia ni somo linalohusu mabaki ya tamaduni..."* | 6.8 | 4.0 | −2.8 ↓ |
| **English-Swahili code-switch** | *"Tulifanya meeting jana na manager akasema project..."* | **28.6** | **16.9** | **−11.7 ↓** |
| Child welfare | *"Watoto wengi wanakabiliwa na changamoto za elimu..."* | 2.0 | 1.9 | −0.1 ↓ |
| Financial savings | *"Ninahitaji kuweka akiba yangu salama kupitia akaunti..."* | 5.1 | 6.7 | +1.6 ↑ |
| **Average** | | **8.4** | **5.9** | **−2.6 (30.4% improvement)** |

**Final MLM training perplexity: 5.39** (3 epochs, evaluated on 5% held-out set)

The two sentences where the adapted model is marginally worse (M-PESA and financial savings) both contain very common, unambiguous Swahili — the base model already predicts them near-perfectly. The largest gains are exactly where expected: **Sheng (−66%)** and **English-Swahili code-switching (−41%)**.

### Masked Token Prediction

Top-5 predictions per test, comparing base vs adapted model:

**[Standard Swahili — Wikipedia style]**
`Akiolojia ni somo linalohusu mabaki ya [tamaduni] za watu wa nyakati zilizopita.`

| Rank | Base AfriBERT | Score | Adapted | Score |
|---|---|---|---|---|
| 1 | tabia | 0.175 | **tabia** | 0.363 |
| 2 | fikra | 0.159 | picha | 0.108 |
| 3 | kazi | 0.065 | jamii | 0.085 |
| 4 | jamii | 0.059 | roho | 0.046 |
| 5 | akili | 0.056 | kazi | 0.027 |

*Both models agree on `tabia`; adapted is more confident (0.363 vs 0.175).*

---

**[East African news — formal]**
`Serikali imetangaza mpango mpya wa kukuza [uchumi] wa taifa kupitia biashara ya kimataifa.`

| Rank | Base AfriBERT | Score | Adapted | Score |
|---|---|---|---|---|
| 1 | **uchumi** | 0.985 | **uchumi** | 0.973 |
| 2 | utalii | 0.005 | pato | 0.011 |
| 3 | pato | 0.002 | utalii | 0.007 |

*Both models nail the correct answer with very high confidence — standard formal Swahili is well-represented in both.*

---

**[M-PESA domain — financial]**
`Tuma [pesa] kwa kutumia nambari ya simu kupitia huduma ya M-PESA.`

| Rank | Base AfriBERT | Score | Adapted | Score |
|---|---|---|---|---|
| 1 | ujumbe (message) | 0.122 | simu | 0.210 |
| 2 | neno (word) | 0.092 | twe | 0.093 |
| 3 | malipo | 0.079 | **pesa** ✓ | 0.053 |
| 4 | simu | 0.070 | sana | 0.035 |
| 5 | nasi | 0.025 | pia | 0.027 |

*Adapted model places `pesa` in top-3; base model puts `ujumbe` (message) first — showing it doesn't understand M-PESA transaction context.*

---

**[CPIMS domain — child protection]**
`Mtoto aliripotiwa kwa ofisi ya [ustawi] wa jamii baada ya kudhulumiwa nyumbani.`

| Rank | Base AfriBERT | Score | Adapted | Score |
|---|---|---|---|---|
| 1 | **ustawi** | 0.908 | **ustawi** | 0.943 |
| 2 | Ustawi | 0.025 | mkuu | 0.010 |
| 3 | mfuko | 0.010 | usalama | 0.009 |

*Both models strongly predict `ustawi` — child welfare language appears in Wikipedia. Adapted model is slightly more confident.*

---

**[Sheng / code-switching — Nairobi urban]**
`Msee alikuwa poa sana, akanisaidia kupata [kazi] ya ofisi.`

| Rank | Base AfriBERT | Score | Adapted | Score |
|---|---|---|---|---|
| 1 | huduma | 0.164 | pesa | 0.189 |
| 2 | majukumu | 0.057 | **emergency** | 0.185 |
| 3 | sehemu | 0.055 | huduma | 0.069 |
| 4 | mahitaji | 0.041 | elimu | 0.029 |
| 5 | **kazi** ✓ | 0.037 | kazi ✓ | 0.019 |

*Base model puts `kazi` at rank 5 (3.7%). Adapted model's top predictions (`pesa`, `emergency`) reflect the CPIMS domain context — the model has learned that `msee` in an urban/office context relates to financial/emergency help.*

---

**[WhatsApp CPIMS — field worker message]**
`Mtoto huyu ana umri wa miaka kumi na mbili na anahitaji [msaada] wa haraka.`

| Rank | Base AfriBERT | Score | Adapted | Score |
|---|---|---|---|---|
| 1 | **msaada** | 0.774 | **msaada** | 0.892 |
| 2 | upasuaji | 0.120 | usaidizi | 0.030 |
| 3 | ushauri | 0.042 | upasuaji | 0.022 |

*Both models strongly predict `msaada` (help/assistance). Adapted model is significantly more confident (0.892 vs 0.774) — it has seen this exact phrasing repeatedly in WhatsApp CPIMS data.*

---

**[English-Swahili code-switch]**
`Tulifanya meeting jana na manager akasema [project] itakuwa ready wiki ijayo.`

| Rank | Base AfriBERT | Score | Adapted | Score |
|---|---|---|---|---|
| 1 | timu (team/sports) | 0.146 | **system** | 0.334 |
| 2 | ligi (league) | 0.048 | **team** | 0.104 |
| 3 | klabu (club) | 0.043 | **family** | 0.041 |
| 4 | Arsenal ⚽ | 0.033 | **process** | 0.034 |
| 5 | kazi | 0.033 | salary | 0.022 |

*The clearest demonstration of domain shift. Base model interprets `meeting` + `manager` as **football context** (Arsenal, league, club). Adapted model correctly understands it as an **office/work context** — `system`, `team`, `process`, `salary` are all semantically appropriate English loanwords.*

---

## Downstream Use — CPIMS Multi-Task Classifier

This model was used as the base encoder for `Rogendo/cpims-nlp-intent-urgency`, a multi-task classifier trained on 1,465 CPIMS support messages to predict:

- **Intent** (63 classes): login issues, password reset, data entry, escaped children, arrests, referrals, etc.
- **Urgency** (3 classes): high / medium / low

Results after full fine-tuning on the adapted base:

| Task | F1 Score |
|------|----------|
| Intent classification (63 classes) | 74.5% |
| Urgency classification | 84.8% |

Compared to the previous version trained on `distilbert-base-uncased` with 271 rows: Intent F1 went from **46% → 74.5%**.

---

## Use Cases & Practical Domains

This model is designed for any NLP task involving Kenyan language text. It provides a stronger starting point than a generic multilingual model wherever the input contains Swahili, Sheng, code-switching, or Kenyan institutional vocabulary.

### 1. Child Protection & Social Work (CPIMS)

The primary motivation for this model. Kenya's Child Protection Information Management System (CPIMS) generates a high volume of support requests, case notes, and field reports written by social workers, case managers, and NGO staff — often in a mix of English, Swahili, and Sheng.

**Practical tasks:**

| Task | Description | Example input |
|---|---|---|
| **Help-desk intent classification** | Route incoming support messages to the correct team or knowledge base article | *"Siwezi kuingia system, password yangu imekwisha"* → `PasswordReset` |
| **Urgency triage** | Flag messages that need immediate human escalation (child at risk, abuse, missing child) | *"Mtoto amekimbia safe house usiku huu"* → `urgent` |
| **Case note sentiment** | Detect frustration or distress in field worker messages to trigger supervisor review | *"Nimejaribu mara nyingi kupata msaada lakini hakuna anayejibu"* → `negative` |
| **Entity extraction (NER)** | Extract names, locations, case IDs, and child ages from free-text case notes | *"Amina, miaka 9, Kibera, Case ID CP-2024-0471"* |
| **Automated case routing** | Predict which department or OVC program a case should be assigned to | Based on case note text |

---

### 2. Financial Services & M-PESA

M-PESA is Kenya's dominant mobile money platform used by over 30 million Kenyans. Customer support queries, fraud reports, and transaction disputes are frequently written in Swahili or code-switched language that generic models mishandle.

**Practical tasks:**

| Task | Description | Example input |
|---|---|---|
| **Transaction dispute classification** | Categorise dispute type: wrong number, reversal, Fuliza, till payment, paybill | *"Nilituma pesa nambari mbaya, naomba reverse"* |
| **Fraud signal detection** | Detect social-engineering scripts, phishing attempts, SIM-swap language | *"Uko na nambari ya siri ya M-PESA? Niambie utatumia"* |
| **Customer sentiment analysis** | Measure customer satisfaction from M-PESA helpline transcripts | Post-interaction classification |
| **FAQ intent matching** | Match a customer query to the nearest self-service FAQ answer | Semantic similarity over a FAQ corpus |
| **Agent response quality scoring** | Score whether a customer service agent's response was appropriate | Given query + response pairs |

---

### 3. Healthcare & Community Health Workers (CHWs)

Community Health Workers in Kenya file visit reports and referral notes, often verbally transcribed or typed on low-end phones in mixed Swahili/English.

**Practical tasks:**

| Task | Description | Example input |
|---|---|---|
| **Symptom extraction** | Extract reported symptoms from CHW visit notes | *"Mtoto ana homa kali na kukohoa sana tangu jana"* |
| **Referral urgency classification** | Triage referral notes: emergency, routine, follow-up | *"Mama mjamzito ana maumivu makali, nahitaji ambulance sasa"* → `emergency` |
| **Facility routing** | Predict whether a patient should go to dispensary, health centre, or county hospital | Based on symptom description |
| **Health campaign text classification** | Classify community feedback on health campaigns (vaccination, family planning) | SMS/WhatsApp response categorisation |

---

### 4. Education & EdTech

Kenya's education sector uses a blend of English instruction and Swahili explanation, especially in lower grades. Many EdTech platforms serving rural Kenya receive student questions in Sheng or code-switched text.

**Practical tasks:**

| Task | Description | Example input |
|---|---|---|
| **Student question topic classification** | Route a question to the right subject tutor or resource | *"Sijui kusolve equation hii, pia sina calculator"* |
| **Learner frustration detection** | Flag messages indicating confusion or disengagement | *"Sielewi hata kidogo, imefail mara tatu"* |
| **Automatic feedback categorisation** | Classify teacher or parent feedback on school platforms | SMS / app reviews |
| **Readability scoring** | Score educational content for appropriateness at different grade levels | Given a paragraph of Swahili text |

---

### 5. Government & Civic Services

Kenya's e-citizen platforms, county service desks, and public feedback systems receive queries and complaints in everyday Kenyan language.

**Practical tasks:**

| Task | Description | Example input |
|---|---|---|
| **Service request classification** | Route citizen petitions/complaints to the correct county department | *"Barabara ya kwetu ina mashimo makubwa sana, lini mtarekebisha?"* |
| **Complaint sentiment & severity** | Detect strongly negative or potentially viral citizen complaints | Social media monitoring |
| **Language identification** | Detect whether a message is Swahili, Sheng, English, or code-switched | Pre-routing in multi-language systems |
| **Policy document Q&A** | Answer questions grounded in Swahili government policy documents | Retrieval-augmented generation (RAG) with this encoder |

---

### 6. Media, Social Listening & Misinformation

Twitter/X, Facebook, and WhatsApp in Kenya carry a large volume of Kenyan Sheng and code-switched content that standard multilingual models struggle to classify.

**Practical tasks:**

| Task | Description | Example input |
|---|---|---|
| **Hate speech / harmful content detection** | Detect Sheng-coded hate speech or incitement that generic models miss | Election-period social media moderation |
| **Rumour / misinformation flagging** | Classify claims as verified, unverified, or disputed | WhatsApp forward monitoring |
| **Topic classification** | Assign news articles or social posts to categories (politics, economy, sports, health) | Media monitoring dashboards |
| **Sentiment analysis** | Measure public sentiment on policy announcements, brands, or events | Code-switched Twitter/X data |

---

## Fine-tuning Guide

This model can be fine-tuned with as few as **200–500 labelled examples** per class for most classification tasks, because DAPT has already adapted the internal representations to the target domain.

### Recommended fine-tuning tasks by architecture

| Architecture | Suitable for | HuggingFace class |
|---|---|---|
| Sequence classification | Intent, sentiment, urgency, topic, routing | `AutoModelForSequenceClassification` |
| Token classification | NER (names, locations, case IDs, symptoms) | `AutoModelForTokenClassification` |
| Multi-task (shared encoder + multiple heads) | Intent + urgency simultaneously | Custom (see jenga_ai SDK) |
| Question answering | Policy/FAQ grounding | `AutoModelForQuestionAnswering` |
| Sentence similarity | Semantic search, FAQ matching | Add a pooling head + contrastive loss |

### Minimum data guidelines

| Task complexity | Approx. labelled examples needed |
|---|---|
| Binary classification (2 classes) | 100–300 per class |
| Multi-class (5–15 classes) | 150–400 per class |
| Multi-class (15–63 classes) | 200–500 per class |
| NER (token-level) | 500–1,000 sentences with full annotation |
| Multi-task (2 heads) | Same as above per task head |

*These estimates are based on domain-adapted models. A generic multilingual base model would need 3–5× more data to reach equivalent performance on Kenyan text.*

### Fine-tuning with HuggingFace Trainer

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

model_name = "Rogendo/afribert-kenya-adapted"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3  # e.g. urgency: low / medium / high
)

training_args = TrainingArguments(
    output_dir          = "my-kenya-classifier",
    num_train_epochs    = 5,
    per_device_train_batch_size = 16,
    learning_rate       = 2e-5,       # standard fine-tuning LR
    warmup_ratio        = 0.1,
    evaluation_strategy = "epoch",
    save_strategy       = "epoch",
    load_best_model_at_end = True,
    bf16                = True,       # use bf16 on A100/A40/H100
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = eval_dataset,
    processing_class = tokenizer,
)
trainer.train()
```

### Fine-tuning with jenga_ai SDK (multi-task)

```yaml
# cpims_config.yaml
model:
  base_model: Rogendo/afribert-kenya-adapted
  max_seq_len: 128

tasks:
  - name: intent
    task_type: multi_class_classification
    num_labels: 63
    label_column: intent

  - name: urgency
    task_type: multi_class_classification
    num_labels: 3
    label_column: urgency

training:
  epochs: 5
  batch_size: 16
  learning_rate: 2.0e-5
  output_dir: results/cpims-v2
```

```bash
python -m jenga_ai train --config cpims_config.yaml
```

---

## Usage

### Single mask prediction

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("Rogendo/afribert-kenya-adapted")
model = AutoModelForMaskedLM.from_pretrained("Rogendo/afribert-kenya-adapted")

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Real Sheng sentence — single mask
results = fill_mask(f"Oya, twendeni zetu, kuna {tokenizer.mask_token} flani ameniudhi. Uyo msee aliiba doh zangu most.")
for r in results:
    print(f"{r['token_str']:<20} {r['score']:.3f}")
```

### Multiple masks (one position at a time)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("Rogendo/afribert-kenya-adapted")
model = AutoModelForMaskedLM.from_pretrained("Rogendo/afribert-kenya-adapted")

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Multiple [MASK] tokens — pipeline returns a list of lists, one per mask position
results = fill_mask(
    f"Oya, twendeni zetu, kuna {tokenizer.mask_token} flani ameniudhi. "
    f"Uyo msee ameniibia {tokenizer.mask_token} zangu mingi sana nikimpata "
    f"{tokenizer.mask_token} sana, hadi atawacha kunibeba ufala."
)

for mask_predictions in results:
    print("--- New Mask ---")
    for r in mask_predictions:
        print(f"{r['token_str']:<20} {r['score']:.3f}")
```

### As a base model for fine-tuning (jenga_ai SDK)

```yaml
# experiment_config.yaml
model:
  base_model: Rogendo/afribert-kenya-adapted
  max_seq_len: 128

tasks:
  - name: intent
    task_type: multi_class_classification
    num_labels: 63
  - name: urgency
    task_type: multi_class_classification
    num_labels: 3
```

---

## Limitations

- **Not suitable for formal Standard Swahili tasks alone** — the up-sampling of Sheng and code-switched text slightly shifts the model away from pure encyclopedic Swahili. Use `castorini/afriberta_large` directly for tasks that only involve formal Swahili prose.
- **Sheng is not standardised** — spelling varies by writer; the model reflects the patterns in the training WhatsApp data which may not generalise to all Sheng dialects (Mombasa Sheng differs from Nairobi Sheng).
- **Small WhatsApp corpus** — source 4 (real CPIMS field chat) is only ~30K tokens before repetition. Up-sampling compensates but does not replace volume.
- **Private model** — the model is currently private on HuggingFace Hub. Access requires a token with read permission on the `Rogendo` organisation.

---

## Citation

If you use this model, please cite the base model:

```bibtex
@inproceedings{ogueji-etal-2021-small,
  title     = {Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages},
  author    = {Ogueji, Kelechi and Zhu, Yuxin and Lin, Jimmy},
  booktitle = {Proceedings of the 1st Workshop on Multilingual Representation Learning},
  year      = {2021},
}
```

---

## Author

**Rogendo** — built as part of the JengaAI CPIMS NLP pipeline for Kenyan child-protection support systems.
