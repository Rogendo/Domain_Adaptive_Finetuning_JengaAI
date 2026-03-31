"""
CPIMS NLP Pipeline — inference module.

Three-step pipeline:
  text input
    → Model: BERT with intent_head (63 classes) + urgency_head (3 classes)
    → Lookup: department router (deterministic map, intent → dept)

Usage:
    from cpims_inference import CPIMSPipeline

    # V2 — HuggingFace Hub (recommended)
    p = CPIMSPipeline(
        model_dir="Rogendo/cpims-nlp-intent-urgency",
        config_path="checkpoints/best/experiment_config.yaml",
    )
    # V1 — local checkpoint (original distilbert run)
    # p = CPIMSPipeline(
    #     model_dir="/results/cpims-nlp/checkpoints/best",
    #     config_path="/root/JengaAI/configs/nlp/cpims_config.yaml",
    # )
    result = p.predict("I cannot log in to CPIMS")
    # {
    #   'intent': 'login',           'intent_confidence': 0.94,
    #   'urgency': 'low',            'urgency_confidence': 0.89,
    #   'department': 'ict_support'
    # }
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from jenga_ai.inference.pipeline import InferencePipeline

# Department routing — deterministic lookup, no model needed
INTENT_TO_DEPT = {
    # case_management
    "ArrestedChildren": "case_management", "Childreninremand": "case_management",
    "Escapeofchildren": "case_management", "ChildSearch": "case_management",
    "ChildOffendorList": "case_management", "ChildTransfer": "case_management",
    "ChildLeave": "case_management", "SelfDischarge": "case_management",
    "ChildDischarge": "case_management", "ChildReaddmission": "case_management",
    "recurringcases": "case_management", "aligningcaseloads": "case_management",
    "SiblingsInfomation": "case_management", "ForeignChildRegistration": "case_management",
    "MissMatchedRequest": "case_management", "complaint": "case_management",
    "genderissues": "case_management", "CPVs": "case_management",
    "crimereport": "case_management", "missinginformation": "case_management",
    # data_registry
    "Admission": "data_registry", "AddmissionDelays": "data_registry",
    "Population": "data_registry", "PopulationUpdate": "data_registry",
    "DataEntry": "data_registry", "DataSubmission": "data_registry",
    "UpdateDetails": "data_registry", "Recordassess": "data_registry",
    "sheets": "data_registry", "personregistry": "data_registry",
    "photos": "data_registry", "delete": "data_registry", "upda": "data_registry",
    "AddStaffTeacher": "data_registry", "Registrationofstaff": "data_registry",
    "addrescuecentre": "data_registry",
    # ict_support
    "login": "ict_support", "passwords": "ict_support",
    "accountunlocking": "ict_support", "UnlockingAccount": "ict_support",
    "createaccount": "ict_support", "createcpmisaccounts": "ict_support",
    "systemAcess": "ict_support", "systemerror": "ict_support",
    "systemfunctions": "ict_support", "Emailaccess": "ict_support",
    "LiveSystem": "ict_support", "LiveSystem-access": "ict_support",
    "ActivateDeactivate-account": "ict_support", "notAbleToReset_Password": "ict_support",
    "CPMISaccess": "ict_support", "CPIMSstatusreport": "ict_support",
    "livestream": "ict_support",
    # training_admin
    "usermanual": "training_admin", "onlinetraining": "training_admin",
    "ChallengesOfStaff": "training_admin", "ManageAdministrativetasks": "training_admin",
    "leavedays": "training_admin", "payments": "training_admin",
    # general
    "greeting": "general", "goodbye": "general", "thanks": "general",
    "help": "general", "about": "general",
}


_REQUIRED_FILES = {"model.pt", "metadata.json", "label_maps.json", "experiment_config.yaml"}


def _resolve_model(model_dir: str, config_path: str) -> tuple[str, str]:
    """Return (local_model_dir, local_config_path).

    Resolution order:
      1. If model_dir is an existing local path AND contains all required
         checkpoint files → use it directly (fast, no download).
      2. Otherwise treat model_dir as a HuggingFace Hub repo ID and
         download it with snapshot_download(), then resolve config_path
         as a relative path inside the downloaded repo.
    """
    from pathlib import Path

    local = Path(model_dir)
    if local.exists():
        missing = [f for f in _REQUIRED_FILES if not (local / f).exists()]
        if not missing:
            local_config = local / "experiment_config.yaml"
            print(f"[CPIMSPipeline] Loading from local path: {local}")
            return str(local), str(local_config)
        print(f"[CPIMSPipeline] Local path {local} is incomplete (missing: {missing}). Falling back to HuggingFace Hub.")

    # Treat as HuggingFace Hub repo ID
    hf_repo = model_dir if not local.exists() else "Rogendo/cpims-nlp-intent-urgency"
    print(f"[CPIMSPipeline] Downloading from HuggingFace Hub: {hf_repo}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is not installed. Run: pip install huggingface_hub")

    repo_local = Path(snapshot_download(repo_id=hf_repo))
    resolved_config = repo_local / config_path
    if not resolved_config.exists():
        resolved_config = repo_local / "experiment_config.yaml"
    if not resolved_config.exists():
        raise FileNotFoundError(
            f"experiment_config.yaml not found in downloaded repo '{hf_repo}'. "
            f"Tried: {repo_local / config_path}"
        )
    local_model_dir = resolved_config.parent
    print(f"[CPIMSPipeline] Using checkpoint: {local_model_dir}")
    return str(local_model_dir), str(resolved_config)


class CPIMSPipeline:
    def __init__(self, model_dir: str, config_path: str):
        """Load the fine-tuned CPIMS model.

        model_dir:   local checkpoint dir OR a HuggingFace Hub repo ID.
                     If the local path does not exist, the model is
                     downloaded from HuggingFace Hub automatically.
        config_path: path to the YAML used during training, or a path
                     relative to the repo root when loading from Hub.
        """
        resolved_dir, resolved_config = _resolve_model(model_dir, config_path)
        self.pipeline = InferencePipeline.from_checkpoint(resolved_dir, resolved_config)

    def predict(self, text: str) -> dict:
        """Run the full three-step pipeline on a single support message."""
        result = self.pipeline.predict(text)

        # Access multi-task output:
        # result.task_results['intent'].heads['intent_head'].label
        # result.task_results['urgency'].heads['urgency_head'].label
        intent_head = result.task_results["intent"].heads["intent_head"]
        urgency_head = result.task_results["urgency"].heads["urgency_head"]

        intent_label = intent_head.label
        intent_conf = intent_head.confidence or 0.0
        urgency_label = urgency_head.label
        urgency_conf = urgency_head.confidence or 0.0
        department = INTENT_TO_DEPT.get(intent_label, "general")

        return {
            "intent": intent_label,
            "intent_confidence": round(intent_conf, 3),
            "urgency": urgency_label,
            "urgency_confidence": round(urgency_conf, 3),
            "department": department,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


if __name__ == "__main__":
    import sys

    # ── Model resolution ──────────────────────────────────────────────────────
    #
    # Pass args to override:
    #   python cpims_inference.py [model_dir] [config_path]
    #
    # Default behaviour (no args):
    #   1. Try local checkpoint first (fast, no download needed)
    #   2. If local not found → download from HuggingFace Hub automatically
    #
    # V1 — distilbert-base-uncased fine-tuned on original 271 rows (local only)
    # model_dir   = "/results/cpims-nlp/checkpoints/best"
    # config_path = "/root/JengaAI/configs/nlp/cpims_config.yaml"
    #
    # V2 — afribert-kenya-adapted fine-tuned on 1,465 augmented rows
    #   Base : Rogendo/afribert-kenya-adapted  (AfriBERT + Swahili/Sheng DAPT)
    #   Model: Rogendo/cpims-nlp-intent-urgency
    #   Intent F1: 74.5%  |  Urgency F1: 84.8%  |  Best checkpoint: epoch 5
    # ─────────────────────────────────────────────────────────────────────────

    # Local path (downloaded from RunPod) — used if it exists, else falls back to Hub
    LOCAL_CKPT  = "/home/naynek/Desktop/CPIMS/sdk_training_runs/results/cpims-nlp-afribert-kenya-adapted/checkpoints/best"
    HF_REPO     = "Rogendo/cpims-nlp-intent-urgency"
    HF_CONFIG   = "checkpoints/best/experiment_config.yaml"

    model_dir   = sys.argv[1] if len(sys.argv) > 1 else LOCAL_CKPT
    config_path = sys.argv[2] if len(sys.argv) > 2 else HF_CONFIG

    pipeline = CPIMSPipeline(model_dir, config_path)

    test_messages = [
        "I cannot log in to CPIMS, the system shows an error",
        "A child has escaped from the rescue centre",
        "How do I add a new staff member to the system?",
        "The population statistics are not updating",
        "Good morning, I need help",
        "The child was arrested last night, we need urgent case entry",
        "My password has expired and I need to reset it",
        "What are the steps for data submission this month?",
    ]

    print(f"\n{'Message':<55} {'Intent':<30} {'Urgency':<8} {'Dept'}")
    print("-" * 115)
    for msg in test_messages:
        r = pipeline.predict(msg)
        short = msg[:52] + "..." if len(msg) > 55 else msg
        print(f"{short:<55} {r['intent']:<30} {r['urgency']:<8} {r['department']}")
