"""


pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-xxx

# Step 1: verify budget estimate
python jenga_gen.py --estimate

# Step 2: dry run to test pipeline (no API calls, no cost)
python jenga_gen.py --run_all --dry_run

# Step 3: start with one domain to verify quality (~$1.33)
python jenga_gen.py --domain mpesa --batches 90

# Step 4: if quality looks good, run remaining domains
python jenga_gen.py --domain ecommerce --batches 75
python jenga_gen.py --domain social --batches 65
python jenga_gen.py --domain banking --batches 55
python jenga_gen.py --domain customer_svc --batches 43

# OR just run everything at once
python jenga_gen.py --run_all







JengaAI Dual-Label Data Generator  v2
=======================================
Budget: $6.00  |  Strategy: dual-label (every call yields MLM + sentiment + urgency)

Every generated example has:
  text        → stripped into MLM corpus  (domain adaptation)
  sentiment   → sentiment fine-tuning     (positive/negative/neutral)
  urgency     → urgency fine-tuning       (urgent/normal/low)

This means one generation run produces THREE training datasets.

Usage:
    python jenga_gen.py --estimate          # show budget breakdown, no API calls
    python jenga_gen.py --domain mpesa --batches 5 --dry_run   # test pipeline
    python jenga_gen.py --run_all           # execute full $6 plan
    python jenga_gen.py --domain mpesa --batches 90            # single domain
"""

import anthropic
import json, time, random, argparse, os
from pathlib import Path
from dataclasses import dataclass

# ─────────────────────────────────────────
# COST CONSTANTS  (Sonnet, March 2026)
# ─────────────────────────────────────────
INPUT_COST_PER_MTK  = 3.00   # $ per 1M input tokens
OUTPUT_COST_PER_MTK = 15.00  # $ per 1M output tokens

# Measured averages for the dual-label prompt below
AVG_INPUT_TOKENS_PER_CALL  = 680
AVG_OUTPUT_TOKENS_PER_CALL = 960   # 12 examples × ~80 tokens each
EXAMPLES_PER_BATCH         = 12

COST_PER_CALL = (
    (AVG_INPUT_TOKENS_PER_CALL  / 1_000_000) * INPUT_COST_PER_MTK  +
    (AVG_OUTPUT_TOKENS_PER_CALL / 1_000_000) * OUTPUT_COST_PER_MTK
)  # ≈ $0.01644


# ─────────────────────────────────────────
# THE $6 GENERATION PLAN
# ─────────────────────────────────────────
# 10% buffer kept for retries  →  328 calls, ~3,936 examples
# Estimated spend: ~$5.39

GENERATION_PLAN = [
    # (domain,          batches,  description)
    ("mpesa",           90,  "M-PESA, mobile money, fraud SMS, agent transactions"),
    ("ecommerce",       75,  "Jumia, Jiji, Kilimall reviews & complaints"),
    ("social",          65,  "Twitter/X, WhatsApp groups, everyday Nairobi life"),
    ("banking",         55,  "KCB, Equity, Tax, Kenya Revenue Authority,Co-op, loans, interest, mobile banking"),
    ("customer_svc",    43,  "Safaricom, bank & delivery customer service calls"),
]


def show_estimate():
    total_calls    = sum(b for _, b, _ in GENERATION_PLAN)
    total_examples = total_calls * EXAMPLES_PER_BATCH
    total_input_tk = total_calls * AVG_INPUT_TOKENS_PER_CALL
    total_output_tk= total_calls * AVG_OUTPUT_TOKENS_PER_CALL
    input_cost     = (total_input_tk  / 1_000_000) * INPUT_COST_PER_MTK
    output_cost    = (total_output_tk / 1_000_000) * OUTPUT_COST_PER_MTK
    total_cost     = input_cost + output_cost

    print("\n" + "═"*60)
    print("  JENGA AI  —  $6 DUAL-LABEL GENERATION PLAN")
    print("═"*60)
    print(f"  {'Domain':<16} {'Calls':>6} {'Examples':>10} {'Cost':>8}")
    print("  " + "─"*54)
    for domain, batches, _ in GENERATION_PLAN:
        ex   = batches * EXAMPLES_PER_BATCH
        cost = batches * COST_PER_CALL
        print(f"  {domain:<16} {batches:>6} {ex:>10,}  ${cost:>6.3f}")
    print("  " + "─"*54)
    print(f"  {'TOTAL':<16} {total_calls:>6} {total_examples:>10,}  ${total_cost:>6.3f}")
    print(f"""
  Each example produces:
    ✓ MLM corpus line          (3,936 lines  ~2.5MB)
    ✓ Sentiment training item  (3,936 labelled)
    ✓ Urgency training item    (3,936 labelled)

  After free public datasets (AfriSenti ~1,800 + MasakhaNER ~3,000):
    Sentiment total  →  ~5,700 examples
    MLM corpus       →  ~3,936 lines synthetic  + Wikipedia + news

  Input tokens  : {total_input_tk:>10,}   cost: ${input_cost:.3f}
  Output tokens : {total_output_tk:>10,}   cost: ${output_cost:.3f}
  ─────────────────────────────────────────────
  ESTIMATED TOTAL : ${total_cost:.2f}  (${6 - total_cost:.2f} buffer remaining)
""")
    print("═"*60 + "\n")


# ─────────────────────────────────────────
# THE DUAL-LABEL SYSTEM PROMPT
# This is the quality lever — do not shorten it
# ─────────────────────────────────────────

DUAL_LABEL_SYSTEM = """You are a Kenyan NLP data specialist. Your task is to generate training data 
for an African language model that understands Kenyan context deeply.

═══ LANGUAGE RULES (non-negotiable) ═══
Write EXACTLY as Kenyans write in digital spaces — WhatsApp, Twitter, Jumia reviews, SMS.

Code-switching patterns to use:
  - Start English, drop to Swahili, finish in Sheng:  "Hii product ni fake kabisa, walinidanganya"
  - Start Sheng, switch mid-sentence:                 "Manze delivery ilichelewa sana, not cool at all"
  - Full Swahili with English brand names:            "Nilituma pesa kwa Paybill lakini haikufika"
  - Gen Z Nairobi Sheng:                              "Si mchezo hii, wameniangusha vibaya sana"
  - English with Kenyan context:                      "Si poa, I ordered a phone on Jumia and it came with a cracked screen. Not happy kabisa."

Sheng vocabulary to weave in naturally:
  manze, sawa, fiti, poa, si poa, kabisa, vibaya, buda, boss, dame, mshene,
  kuingia box, kushoot, fire (=great), fresh (=good/authentic), ngori (=bad), noma (=bad), keja (=home), jasho (=sweat = hard work), 
  doh/mkwanja (=money), kuomoka/kutoka block (=succeed), kushuka (=go down/cheap), rada (=network)

M-PESA / payment vocabulary:
  tuma, piga M-PESA, lipa na M-PESA, withdraw, top up, float, till number,
  paybill, agent wa M-PESA, hakuna float, transaction imefail, reversal

Kenyan institutions / brands to reference:
  Safaricom, M-PESA, Airtel Money, KCB, Equity Bank, Co-op Bank, NCBA,
  KRA, iTax, NHIF, NSSF, Huduma Namba, Jumia Kenya, Jiji, Kilimall,
  Tala, Branch, Fuliza, M-Shwari, KCB M-PESA

Kenyan places / context:
  CBD, Westlands, Eastlands, Kibera, Karen, Ngong Road, Thika Road,
  Mombasa, Kisumu, Nakuru, Eldoret, matatu, boda boda, stage

═══ LABEL DEFINITIONS ═══

SENTIMENT:
  positive  — satisfaction, praise, would recommend, happy with outcome
  negative  — frustration, complaint, disappointment, warning others
  neutral   — mixed feelings, purely factual, ambiguous, information-seeking

URGENCY:
  urgent    — money lost NOW, account compromised NOW, fraud happening NOW,
              deadline in hours, health emergency, system down mid-transaction
  normal    — general complaint or inquiry, issue that can wait, routine request
  low       — casual feedback, future planning, information request, general praise

═══ QUALITY RULES ═══
- Text length: 1–4 sentences (vary this — some short, some longer)
- Each batch: aim for ~35% negative sentiment, ~40% positive, ~25% neutral
- Each batch: aim for ~20% urgent, ~50% normal, ~30% low urgency
- Urgent + negative often co-occur — but not always (e.g. urgent + neutral = factual emergency)
- Make the text FEEL real — include amounts (Ksh 500, elfu tano), dates, product names
- Avoid generic English — that is not what this model needs

═══ OUTPUT FORMAT (strict) ═══
Return ONLY a valid JSON array. No preamble. No markdown fences. No explanation.
Each object must have exactly these fields:
{
  "text": "the message in natural Kenyan language",
  "sentiment": "positive|negative|neutral",
  "urgency": "urgent|normal|low",
  "language_mix": "sheng_dominant|swahili_dominant|english_dominant|mixed",
  "domain": "the domain context"
}"""


# ─────────────────────────────────────────
# USER PROMPTS PER DOMAIN
# ─────────────────────────────────────────

DOMAIN_PROMPTS = {
    "mpesa": """Generate 12 messages from Kenyan M-PESA users.

Cover a mix of situations:
- Sending/receiving money to family or business
- Paying bills via Paybill or Till Number (KPLC, Nairobi Water, rent)
- Fuliza overdraft — relief, complaint about interest, payback stress
- M-Shwari savings and loans — satisfaction, failure, interest complaints
- Agent issues — no float, agent gave wrong change, far agent location
- Transaction failed — money deducted but not received, reversal pending
- Fraud SMS received — fake Safaricom, fake prize, fake reversal request
- Complimenting M-PESA convenience for business

Remember: vary sentiment AND urgency across the 12 examples.
Domain tag: "mpesa"
""",

    "ecommerce": """Generate 12 messages from Kenyan online shoppers.

Cover a mix of situations:
- Product reviews: phones, clothes, shoes, electronics, household items on Jumia/Jiji
- Delivery experience: arrived on time, late, wrong item, damaged packaging
- Seller behaviour: responsive, ghosting after payment, fake product sent
- Pricing: bargaining on Jiji, price differences between platforms, flash sales
- Return and refund process: successful, frustrating, unresolved
- Comparing Jumia vs Jiji vs Kilimall experience
- Counterfeit or "fake" products discovered after delivery
- Shop owner or seller perspective: customer complaints received, demand patterns

Domain tag: "ecommerce"
""",

    "social": """Generate 12 social media posts from Nairobi residents.

Cover a mix of topics:
- Cost of living complaints: unga, cooking gas, electricity bill (KPLC), fuel
- Transport: matatu fare hike, Bolt/Uber surge pricing, boda boda accidents, SGR
- Government and politics: KRA taxes, Huduma services, county services, corrupt officials
- Everyday wins: small business success, promotion at work, family milestone
- Entertainment: Kenyan music, local football, Netflix, TikTok trends
- Tech and mobile: Safaricom network, app crashes, new M-PESA features
- Weather and environment: Nairobi flooding, drought, load shedding

Write in authentic Gen Z Nairobi voice. Heavy Sheng expected.
Domain tag: "social"
""",

    "banking": """Generate 12 messages from Kenyan bank customers.

Cover a mix of situations:
- Mobile banking app issues: KCB app, Equity Mobile, Co-op app — crashes, failed transfers
- Loan applications: approved, rejected, interest rate shock, repayment stress
- Account statements: unexpected charges, ledger fees, SMS fees
- ATM issues: card swallowed, ATM out of cash, double deduction
- Customer service quality: helpful agent vs rude agent vs long hold times
- Interest rates: savings account rates, fixed deposit, comparing banks
- Fraud and security: unauthorized transaction, account suspended for suspicious activity
- Cheque clearing delays, RTGS transfers, inter-bank transfers

Domain tag: "banking"
""",

    "customer_svc": """Generate 12 messages representing customer service interactions.

These are messages a customer SENDS TO or ABOUT a service provider — 
Safaricom call centre, bank customer service, Jumia support, delivery company.

Cover a mix of situations:
- Customer explaining their problem to support (first contact)
- Customer escalating after issue not resolved first time
- Customer satisfied after issue resolved — praising agent
- Customer frustrated after multiple failed contacts
- Customer reporting a specific agent by name (good or bad)
- Post-resolution feedback — did the fix actually work?
- Waiting on hold / chat frustration
- IVR / automated system complaints ("press 1 for Swahili is not working")

Write from the customer's perspective, in natural Kenyan language.
Domain tag: "customer_svc"
""",
}


# ─────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────

class DualLabelGenerator:

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        if not dry_run:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Set ANTHROPIC_API_KEY environment variable")
            self.client = anthropic.Anthropic(api_key=key)

        # Output dirs
        for d in ["./generated/mlm", "./generated/tasks", "./generated/logs"]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Session stats
        self.calls = 0
        self.examples = 0
        self.failed = 0
        self.input_tokens = 0
        self.output_tokens = 0

    # ── API call ──────────────────────────────────
    def _call(self, domain: str) -> list[dict]:
        if self.dry_run:
            return [
                {"text": f"[DRY RUN {domain}] example {i}",
                 "sentiment": random.choice(["positive","negative","neutral"]),
                 "urgency":   random.choice(["urgent","normal","low"]),
                 "language_mix": "mixed",
                 "domain": domain}
                for i in range(EXAMPLES_PER_BATCH)
            ]

        resp = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1800,
            system=DUAL_LABEL_SYSTEM,
            messages=[{"role": "user", "content": DOMAIN_PROMPTS[domain]}]
        )

        self.input_tokens  += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        self.calls += 1

        raw = resp.content[0].text.strip()
        raw = raw.replace("```json","").replace("```","").strip()

        examples = json.loads(raw)

        # Validate schema — drop malformed items silently
        required = {"text","sentiment","urgency","language_mix","domain"}
        valid = [e for e in examples if required.issubset(e.keys())]

        # Enforce label validity
        for e in valid:
            if e["sentiment"] not in ("positive","negative","neutral"):
                e["sentiment"] = "neutral"
            if e["urgency"] not in ("urgent","normal","low"):
                e["urgency"] = "normal"

        self.examples += len(valid)
        return valid

    # ── Run batches for one domain ─────────────────
    def generate_domain(self, domain: str, batches: int,
                        delay: float = 0.5) -> list[dict]:
        all_examples = []
        print(f"\n  [{domain.upper()}]  {batches} batches  "
              f"(~{batches * EXAMPLES_PER_BATCH} examples  ~${batches * COST_PER_CALL:.2f})")

        for i in range(batches):
            try:
                batch = self._call(domain)
                all_examples.extend(batch)

                if (i + 1) % 15 == 0 or i == batches - 1:
                    cost_now = (
                        (self.input_tokens  / 1e6 * INPUT_COST_PER_MTK) +
                        (self.output_tokens / 1e6 * OUTPUT_COST_PER_MTK)
                    )
                    print(f"    {i+1:>3}/{batches}  |  "
                          f"total examples: {self.examples:>5,}  |  "
                          f"spent: ${cost_now:.4f}")

                time.sleep(delay)

            except json.JSONDecodeError:
                self.failed += 1
                print(f"    ⚠ parse error batch {i+1} — skipping")
                time.sleep(1.0)

            except anthropic.RateLimitError:
                print(f"    ⏳ rate limit — waiting 20s...")
                time.sleep(20)
                try:
                    batch = self._call(domain)
                    all_examples.extend(batch)
                except Exception as e:
                    self.failed += 1
                    print(f"    ✗ retry failed: {e}")

            except Exception as e:
                self.failed += 1
                print(f"    ✗ error batch {i+1}: {type(e).__name__}: {e}")
                time.sleep(2.0)

        print(f"  ✓ {domain}: {len(all_examples)} examples collected")
        return all_examples

    # ── Save: splits each into MLM + sentiment + urgency ──
    def save(self, examples: list[dict], domain: str):
        """
        From one list of dual-labelled examples, write:
          1. MLM corpus  — text only, appended to master file
          2. Sentiment   — text + sentiment label
          3. Urgency     — text + urgency label
        All task files are JSONL, split 80/10/10.
        """
        if not examples:
            return

        random.shuffle(examples)
        n = len(examples)
        train_end = int(n * 0.8)
        val_end   = int(n * 0.9)
        splits = {
            "train": examples[:train_end],
            "val":   examples[train_end:val_end],
            "test":  examples[val_end:]
        }

        # 1. MLM corpus (text only)
        mlm_path = "./generated/mlm/master_mlm_corpus.txt"
        with open(mlm_path, "a", encoding="utf-8") as f:
            for ex in examples:
                text = ex["text"].strip()
                if text and len(text) > 15:
                    f.write(text + "\n")

        # 2. Sentiment JSONL
        for split, data in splits.items():
            path = f"./generated/tasks/sentiment_{split}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                for ex in data:
                    f.write(json.dumps({
                        "text":         ex["text"],
                        "label":        ex["sentiment"],
                        "language_mix": ex["language_mix"],
                        "domain":       ex["domain"],
                        "source":       "synthetic_claude"
                    }, ensure_ascii=False) + "\n")

        # 3. Urgency JSONL
        for split, data in splits.items():
            path = f"./generated/tasks/urgency_{split}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                for ex in data:
                    f.write(json.dumps({
                        "text":         ex["text"],
                        "label":        ex["urgency"],
                        "language_mix": ex["language_mix"],
                        "domain":       ex["domain"],
                        "source":       "synthetic_claude"
                    }, ensure_ascii=False) + "\n")

        mlm_lines = n
        print(f"  → Saved {domain}: "
              f"MLM +{mlm_lines} | "
              f"sentiment {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} | "
              f"urgency {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    # ── Print session summary ──────────────────────
    def print_stats(self):
        i_cost = (self.input_tokens  / 1e6) * INPUT_COST_PER_MTK
        o_cost = (self.output_tokens / 1e6) * OUTPUT_COST_PER_MTK
        total  = i_cost + o_cost

        # Count output files
        mlm_path = Path("./generated/mlm/master_mlm_corpus.txt")
        mlm_lines = sum(1 for _ in open(mlm_path)) if mlm_path.exists() else 0
        mlm_mb    = mlm_path.stat().st_size / 1e6 if mlm_path.exists() else 0

        sent_train = Path("./generated/tasks/sentiment_train.jsonl")
        sent_count = sum(1 for _ in open(sent_train)) if sent_train.exists() else 0

        urg_train  = Path("./generated/tasks/urgency_train.jsonl")
        urg_count  = sum(1 for _ in open(urg_train)) if urg_train.exists() else 0

        print("\n" + "═"*55)
        print("  SESSION COMPLETE")
        print("═"*55)
        print(f"  API calls           : {self.calls:,}")
        print(f"  Failed calls        : {self.failed:,}")
        print(f"  Total examples      : {self.examples:,}")
        print(f"  Input tokens        : {self.input_tokens:,}  (${i_cost:.4f})")
        print(f"  Output tokens       : {self.output_tokens:,}  (${o_cost:.4f})")
        print(f"  ─────────────────────────────────────────")
        print(f"  TOTAL SPENT         : ${total:.4f}")
        print(f"  BUDGET REMAINING    : ${6.00 - total:.4f}")
        print(f"\n  OUTPUT FILES:")
        print(f"  MLM corpus          : {mlm_lines:,} lines  ({mlm_mb:.2f}MB)")
        print(f"  Sentiment train     : {sent_count:,} examples")
        print(f"  Urgency train       : {urg_count:,} examples")
        print("═"*55 + "\n")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JengaAI $6 Dual-Label Generator")
    parser.add_argument("--estimate",  action="store_true",
                        help="Show budget breakdown and exit")
    parser.add_argument("--run_all",   action="store_true",
                        help="Execute the full $6 generation plan")
    parser.add_argument("--domain",    default=None,
                        help=f"Single domain: {list(DOMAIN_PROMPTS.keys())}")
    parser.add_argument("--batches",   type=int, default=10,
                        help="Batches for single-domain run")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Test pipeline without API calls")
    args = parser.parse_args()

    if args.estimate:
        show_estimate()
        return

    gen = DualLabelGenerator(dry_run=args.dry_run)

    if args.run_all:
        show_estimate()
        print("Starting full generation plan...\n")
        for domain, batches, desc in GENERATION_PLAN:
            print(f"  {desc}")
            examples = gen.generate_domain(domain, batches)
            gen.save(examples, domain)
        gen.print_stats()

    elif args.domain:
        if args.domain not in DOMAIN_PROMPTS:
            print(f"Unknown domain '{args.domain}'. "
                  f"Available: {list(DOMAIN_PROMPTS.keys())}")
            return
        examples = gen.generate_domain(args.domain, args.batches)
        gen.save(examples, args.domain)
        gen.print_stats()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
