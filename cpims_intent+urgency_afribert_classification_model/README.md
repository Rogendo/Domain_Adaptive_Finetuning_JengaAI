---
language:
  - sw
  - en
tags:
  - text-classification
  - multi-task-learning
  - intent-classification
  - urgency-classification
  - cpims
  - kenya
  - swahili
  - sheng
  - child-protection
  - jenga-ai
base_model: Rogendo/afribert-kenya-adapted
license: apache-2.0
---

# CPIMS Intent + Urgency Classifier

`Rogendo/cpims-nlp-intent-urgency` is a multi-task NLP classifier built for the **Child Protection Information Management System (CPIMS)** in Kenya. It simultaneously predicts the **intent** of a support message (63 classes) and its **urgency level** (high / medium / low) from a single pass through a shared encoder.

---

## Model Architecture

The model uses a **shared encoder + dual classification head** design:

```
Input text
    ↓
[Rogendo/afribert-kenya-adapted]   ← XLM-RoBERTa encoder, domain-adapted for Kenyan language
    ↓
[CLS] representation
    ├──→ intent_head   (Linear → 63 classes)   weight = 1.0
    └──→ urgency_head  (Linear → 3 classes)    weight = 0.5
```

Both heads share the same encoder weights. Training jointly learns that *"I cannot log into CPIMS"* is simultaneously a **`login`** intent and **`low`** urgency from the same token representations — more efficient than training two separate models.

| Component | Detail |
|---|---|
| Base encoder | `Rogendo/afribert-kenya-adapted` (AfriBERT, 125.7M params) |
| Frozen layers | Bottom 6 of 12 encoder layers (reduces overfit on small dataset) |
| Intent head | Linear(hidden\_size → 63), dropout 0.2 |
| Urgency head | Linear(hidden\_size → 3), dropout 0.2 |
| Framework | [jenga\_ai SDK](https://github.com/TujengeAI/JengaAI) |

---

## Training Data

### Base dataset — `cpims_training.csv`

271 labelled CPIMS support messages collected and deduplicated from two internal sources:

- `data.json` — intent patterns from the original CPIMS virtual assistant rule set
- `INTENTS.json` — annotated support ticket intents used for chatbot training

Each row has: `text`, `intent` (63 classes), `urgency` (high/medium/low), `department`.

### Augmented dataset — `cpims_training_augmented.csv` (used for training)

The base 271 rows were augmented to **1,465 rows** using two pipelines:

**1. WhatsApp chat augmentation** (`augment_whatsapp.py`)**

Three real WhatsApp support group exports from CPIMS field workers were parsed:

| File | Description |
|---|---|
| `whatsappchat-Msa.txt` | Mombasa CPIMS field worker group |
| `whatsappchat-Nairobi.txt` | Nairobi CPIMS field worker group |
| `whatsappchat-Wajiri.txt` | Wajiri region CPIMS support group |

Messages were filtered (>20 characters, no media), then auto-labelled using 63 keyword-matching intent rules. This added 1,194 new rows of real conversational field worker language.

**2. Synthetic data generation** (`generate_synthetic.py`)**

For intent classes with fewer than 20 examples in the augmented set, template-based synthetic samples were generated covering English, Swahili, and Kenyan code-switching. This targeted 47 underrepresented classes.

### Final data split

| Split | Rows | % |
|---|---|---|
| Train | ~1,245 | 85% |
| Validation | ~220 | 15% |

`test_size: 0.15`, `seed: 42` — stratified by intent label.

---

## Training Configuration

Trained using the **jenga\_ai SDK** with the following `cpims_intent_urgency.yaml`:

```yaml
project_name: cpims_nlp_pipeline

model:
  base_model: Rogendo/afribert-kenya-adapted
  dropout: 0.2
  freeze_encoder_layers: 6      # freeze bottom 6/12 layers
  gradient_checkpointing: false

tokenizer:
  max_length: 64                # CPIMS messages are short (SMS/chat style)
  padding: max_length
  truncation: true

training:
  learning_rate: 3.0e-5
  batch_size: 8
  num_epochs: 20
  weight_decay: 0.01
  warmup_steps: 50
  max_grad_norm: 1.0
  early_stopping_patience: 5
  metric_for_best_model: eval_loss

tasks:
  - name: intent
    type: single_label_classification
    text_column: text
    label_column: intent
    heads:
      - name: intent_head
        num_labels: 63
        weight: 1.0
        dropout: 0.2

  - name: urgency
    type: single_label_classification
    text_column: text
    label_column: urgency
    heads:
      - name: urgency_head
        num_labels: 3
        weight: 0.5             # urgency weighted less than intent
        dropout: 0.2
```

**Hardware:** CPU (RunPod) — no GPU required for fine-tuning at this scale with frozen encoder layers.

**Early stopping:** Best checkpoint saved at **epoch 5** out of 20 (patience=5 on eval loss).

---

## Performance

### Best checkpoint — Epoch 5

| Task | Accuracy | F1 Score |
|---|---|---|
| Intent (63 classes) | 74.5% | 70.5% |
| Urgency (3 classes) | 85.0% | 84.8% |

### Comparison to previous version

The previous CPIMS classifier was trained on `distilbert-base-uncased` with only 271 rows:

| Version | Base model | Training rows | Intent F1 |
|---|---|---|---|
| V1 | `distilbert-base-uncased` | 271 | 46.0% |
| **V2 (this model)** | `Rogendo/afribert-kenya-adapted` | 1,465 | **74.5%** |

**+28.5 percentage points** improvement — driven by both the domain-adapted Kenyan base model and the augmented training data.

---

## Intent Classes (63)

All intents correspond to real CPIMS support scenarios covering system access, case management, child welfare operations, and general enquiries:

| Category | Intents |
|---|---|
| **Account & Access** | `login`, `passwords`, `UnlockingAccount`, `accountunlocking`, `notAbleToReset_Password`, `ActivateDeactivate-account`, `systemAcess`, `CPMISaccess`, `Emailaccess`, `createaccount`, `createcpmisaccounts` |
| **Child Case Management** | `DataEntry`, `DataSubmission`, `UpdateDetails`, `missinginformation`, `ChildSearch`, `ChildTransfer`, `ChildDischarge`, `ChildReaddmission`, `SiblingsInfomation`, `recurringcases`, `aligningcaseloads` |
| **Child Welfare & Safety** | `ArrestedChildren`, `Escapeofchildren`, `Childreninremand`, `ChildOffendorList`, `ChildLeave`, `SelfDischarge`, `genderissues`, `crimereport` |
| **Registration & Admissions** | `Admission`, `AddmissionDelays`, `ForeignChildRegistration`, `Registrationofstaff`, `AddStaffTeacher`, `personregistry`, `Population`, `PopulationUpdate` |
| **System & Technical** | `systemerror`, `systemfunctions`, `LiveSystem`, `LiveSystem-access`, `livestream`, `CPIMSstatusreport` |
| **Administration** | `ManageAdministrativetasks`, `ChallengesOfStaff`, `addrescuecentre`, `leavedays`, `payments`, `photos`, `sheets`, `usermanual`, `onlinetraining`, `delete`, `upda` |
| **General Conversation** | `greeting`, `goodbye`, `help`, `thanks`, `about`, `complaint`, `CPVs`, `Recordassess` |

### Urgency Classes (3)

| Label | Meaning | Example |
|---|---|---|
| `high` | Child at immediate risk — escalate now | *"Mtoto amekimbia safe house usiku huu"* |
| `medium` | Needs resolution today — assign to team | *"Nimejaribu kuingia mara nyingi, account imefungwa"* |
| `low` | Routine request — can be queued | *"Asante sana kwa msaada wako wa leo"* |

---

## Usage

### jenga\_ai SDK (recommended)

```python
from jenga_ai.inference.pipeline import InferencePipeline

pipeline = InferencePipeline.from_checkpoint(
    model_dir="Rogendo/cpims-nlp-intent-urgency",
    config_path="checkpoints/best/experiment_config.yaml",
)

result = pipeline.predict("Nimesahau password, tafadhali nisaidie")
print(result)
# intent:  passwords  (0.999)
# urgency: medium     (0.997)
```

### Batch inference

```python
texts = [
    "Nimesahau password, tafadhali nisaidie",
    "Mtoto amekimbia kutoka kituo chetu, ninaomba msaada wa haraka",
    "I cannot log into CPIMS, the system shows an error",
    "How do I add a new staff member to the system",
    "Asante sana kwa msaada wako wa leo",
]

for text in texts:
    result = pipeline.predict(text)
    tasks = result.get("task_results", {})
    intent  = tasks.get("intent",  {}).get("heads", {}).get("intent_head",  {})
    urgency = tasks.get("urgency", {}).get("heads", {}).get("urgency_head", {})
    print(f"\nText   : {text}")
    print(f"Intent : {intent.get('label')}  ({intent.get('confidence', 0):.3f})")
    print(f"Urgency: {urgency.get('label')} ({urgency.get('confidence', 0):.3f})")
```

### HuggingFace Transformers (raw)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Note: for the full multi-task output, use the jenga_ai pipeline above.
# The transformers API exposes a single-head view.
tokenizer = AutoTokenizer.from_pretrained("Rogendo/cpims-nlp-intent-urgency")
inputs = tokenizer("Nimesahau password", return_tensors="pt", max_length=64, truncation=True)
```

---

## Training Pipeline — Two-Stage

This model is the **second stage** of a two-stage Kenyan NLP pipeline:

```
Stage 1 — Domain-Adaptive Pre-Training (MLM)
  castorini/afriberta_large
        ↓  (continued MLM on ~39M Kenyan tokens)
  Rogendo/afribert-kenya-adapted
        ↓
Stage 2 — Multi-Task Fine-tuning (Classification)
  Rogendo/afribert-kenya-adapted
        ↓  (1,465 CPIMS messages, 20 epochs, early stopping at epoch 5)
  Rogendo/cpims-nlp-intent-urgency  ← this model
```

Stage 1 training data:
- Swahili Wikipedia (`wikimedia/wikipedia`, 20231101.sw) — ~22M tokens
- MasakhaNEWS East African journalism (`masakhane/masakhanews`, swa) — ~1M tokens ×3
- Synthetic Sheng/code-switch corpus — ~1M tokens ×10
- Real CPIMS WhatsApp field chat — ~30K tokens ×20

---

## Languages

The model handles all three registers common in Kenyan institutional communication:

| Language | Example |
|---|---|
| English | *"I cannot log into CPIMS, the system shows an error"* |
| Swahili | *"Mtoto aliripotiwa kwa ofisi ya ustawi wa jamii"* |
| Code-switch (Sheng/Kiswanglish) | *"Nimesahau password, tafadhali nisaidie"* |

---

## Limitations

- **63-class intent space is wide** — some low-frequency classes (e.g. `CPVs`, `Recordassess`, `upda`) had only 1–2 training examples even after augmentation. Predictions for these classes may be unreliable.
- **Short message optimised** — `max_length: 64`. Very long messages (>64 tokens) are truncated; important context near the end of long case notes may be lost.
- **Urgency is rule-approximated** — urgency labels in training data were assigned by keyword rules, not by human clinical judgement. High-stakes urgency decisions should always be reviewed by a trained social worker.
- **Private model** — currently private on HuggingFace Hub. Access requires a read token on the `Rogendo` organisation.

---

## Citation

If you use this model, please cite the base models:

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

Model trained using the [jenga\_ai SDK](https://github.com/TujengeAI/JengaAI) — an open-source multi-task NLP training framework for African languages.
