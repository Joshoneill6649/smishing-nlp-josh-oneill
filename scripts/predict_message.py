#!/usr/bin/env python3
# 1) Loads the trained fusion model:
#       models/emotion_fusion_model_1.joblib
#       reports/emotion_fusion_model_1_report.json
#
# 2) Uses backbone models:
#       models/bert_preproc256_v1/
#       joeddav/distilbert-base-uncased-go-emotions-student
#
# 3) Asks message via command line 
#
# 4) Prints:
#       smish vs ham probabilities from the fusion model
#       the final decision using the saved threshold
#       the top three GoEmotions plus an Other remainder
#       a simple user profile risk interpretation from user_profiles.py

from __future__ import annotations
from pathlib import Path
import json
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# Section: fixed configuration and paths
ROOT = Path(".")
FUSION_MODEL_PATH  = ROOT / "models" / "emotion_fusion_model_1.joblib"
FUSION_REPORT_PATH = ROOT / "reports" / "emotion_fusion_model_1_report.json"
BERT_DIR           = ROOT / "models" / "bert_preproc256_v1"
EMO_MODEL_NAME     = "joeddav/distilbert-base-uncased-go-emotions-student"

EMO_MAX_LEN = 128
BERT_MAX_LEN = 256

# Add local scripts directory to sys.path so we can import project utilities
sys.path += [str(ROOT), str(ROOT / "scripts")]

# Section: device selection
# Choose the best available device in order: cuda, then mps, then cpu
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

DEVICE = pick_device()

# Section: text normaliser uses text_preprocess.py 
try:
    from scripts.text_preprocess import normalize_text  # type: ignore
except Exception:
    try:
        from text_preprocess import normalize_text  # type: ignore
    except Exception:
        def normalize_text(s: str) -> str:
            return str(s).strip()

# Section: user profiles 
# Try to import assess_all_profiles from user_profiles.py; otherwise disable profile view
try:
    from user_profiles import assess_all_profiles  # type: ignore
except Exception:
    assess_all_profiles = None

# Section: helper functions

# Run the GoEmotions backbone
# Input: list of texts
# Output: numpy array of shape [N, L] with per label probabilities
@torch.no_grad()
def emo_probs(texts, tok, model) -> np.ndarray:
    enc = tok(
        texts,
        truncation=True,
        padding=True,
        max_length=EMO_MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs

# Run the BERT smishing backbone
# Input: list of texts
# Output: numpy array of shape [N] with probability of class 1 (smish)
@torch.no_grad()
def bert_pos_prob(texts, tok, model) -> np.ndarray:
    enc = tok(
        texts,
        truncation=True,
        padding=True,
        max_length=BERT_MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    return probs

# Load the decision threshold from the fusion model report
# Returns 0.5 if the threshold is missing
def load_threshold(report_path: Path) -> float:
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(report_path.read_text())
        thr = data.get("threshold", None)
        if isinstance(thr, (int, float)):
            return float(thr)
    return 0.5

# For one emotion probability vector:
# 1) Return the top k (label, probability) pairs
# 2) Compute Other as one minus the sum of the top k probabilities
def top_k_with_other(prob_row: np.ndarray, id2label, k=3):
    pairs = [(id2label[i], float(prob_row[i])) for i in range(len(prob_row))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    topk = pairs[:k]
    other = max(0.0, 1.0 - sum(p for _, p in topk))
    return topk, other

# Format a float in the range zero to one as a percentage string
def pct(x: float) -> str:
    return f"{100.0 * max(0.0, min(1.0, x)):5.1f}%"

# A small text bar for quick visual display of a probability
def bar(x: float, width: int = 24) -> str:
    x = max(0.0, min(1.0, x))
    n = int(round(x * width))
    return "█" * n + " " * (width - n)

# Section: input helper for a single message
# Read one message from either piped standard input or interactive input
def read_message() -> str:
    # Non interactive or piped input: read everything and strip
    try:
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            return data.strip()
    except Exception:
        pass

    # Interactive mode: prompt once and read a single line
    msg = input("Enter message: ")
    return msg.strip()

# Section: main logic

def main():
    # Step 1: check that required artifacts exist
    if not FUSION_MODEL_PATH.exists():
        raise SystemExit(f"Missing fusion model: {FUSION_MODEL_PATH}")
    if not FUSION_REPORT_PATH.exists():
        raise SystemExit(f"Missing fusion report: {FUSION_REPORT_PATH}")
    if not BERT_DIR.exists():
        raise SystemExit(f"Missing BERT dir: {BERT_DIR}")

    # Step 2: load fusion model and its decision threshold
    pack = joblib.load(FUSION_MODEL_PATH)
    clf = pack["model"]
    thr = load_threshold(FUSION_REPORT_PATH)

    # Step 3: load emotion and BERT backbones
    emo_tok = AutoTokenizer.from_pretrained(EMO_MODEL_NAME)
    emo_mod = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL_NAME).to(DEVICE)

    bert_tok = AutoTokenizer.from_pretrained(BERT_DIR)
    bert_mod = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(DEVICE)

    # Build ordered list of emotion labels so index maps to label name
    id2label_map = getattr(emo_mod.config, "id2label", {i: f"emo_{i}" for i in range(emo_mod.config.num_labels)})
    id2label = [id2label_map[i] for i in range(len(id2label_map))]

    # Step 4: read message from user
    msg = read_message()
    if not msg:
        raise SystemExit("No input text provided.")

    text = normalize_text(msg)

    # Step 5: build features for the fusion model
    # e is the emotion probability vector
    # b is the BERT smish probability
    e = emo_probs([text], emo_tok, emo_mod)[0]
    b = bert_pos_prob([text], bert_tok, bert_mod)[0]

    # Concatenate into a single feature vector: all emotion probabilities plus the BERT probability
    feats = np.concatenate([e.astype(np.float32), np.array([b], dtype=np.float32)], axis=0)
    X = feats.reshape(1, -1)

    # Step 6: fusion model prediction
    proba = clf.predict_proba(X)[0]
    p_ham = float(proba[0])
    p_smish = float(proba[1])
    label = "SMISH" if p_smish >= thr else "HAM"

    # Step 7: emotion summary (top three plus Other)
    top3, other = top_k_with_other(e, id2label, k=3)

    # Step 8: print the core results
    print("\n================ RESULT ================")
    print(f"Message           : {text}")
    print(f"Predicted label   : {label}")
    print(f"Smish probability : {pct(p_smish)}  {bar(p_smish)}")
    print(f"Ham probability   : {pct(p_ham)}  {bar(p_ham)}")
    print(f"Decision threshold: {thr:.6f}")

    print("\nTop emotions (GoEmotions backbone)")
    for i, (name, prob) in enumerate(top3, start=1):
        print(f"  {i}. {name:<20} {pct(prob)}  {bar(prob)}")
    print(f"  └─ Other              {pct(other)}  {bar(other)}")

    # Step 9: user profile risk overlay (optional, no side effects if missing)
    if assess_all_profiles is not None:
        emo_dict = {id2label[i]: float(e[i]) for i in range(len(id2label))}
        profiles = assess_all_profiles(p_smish, emo_dict)

        print("\nUser profile risk interpretation")
        for p in profiles:
            name = p.get("name", "Unknown")
            risk_label = p.get("risk_label", "N/A")
            score = float(p.get("risk_score", 0.0))
            why = p.get("why", "")
            print(f"  - {name:22} {risk_label:<6} (score={score:.2f})  {why}")
    else:
        print("\n[info] user_profiles.py not found, skipping profile based view.")

    print("========================================")

if __name__ == "__main__":
    main()
