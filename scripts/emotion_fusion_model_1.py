#!/usr/bin/env python3
# scripts/emotion_fusion_model_1.py
#
# Late fusion: combine GoEmotions probabilities with a local BERT smishing score using Logistic Regression.
# Train on train_balanced, pick one F1-max threshold on val_deploy, then evaluate on test and test_deploy.
# Backbones are frozen: GoEmotions model from Hugging Face and the best local BERT folder.
#
# INPUTS
# data/processed/train_balanced.parquet
# data/processed/val_deploy.parquet
# data/processed/test.parquet
# data/processed/test_deploy.parquet
# models/bert_preproc256_v1  or  artifacts/bert_preproc256_v1   (local best BERT)
#
# OUTPUTS
# models/emotion_fusion_model_1.joblib
# reports/emotion_fusion_model_1_report.json
# reports/predictions_emotion_fusion_model_1_{val_deploy,test,test_deploy}_with_probs.csv
# reports/csv/baseline_train_log.csv                    (appended)
# reports/csv/baseline_metrics_log.csv                  (appended)

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json, sys, random
import numpy as np
import pandas as pd
import joblib

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

# Fixed configuration for reproducibility and naming
SEED = 1337
TAG  = "emotion_fusion_model_1"

# Canonical data paths
ROOT = Path(".")
PROC = ROOT/"data"/"processed"
TRAIN_PATH = PROC/"train_balanced.parquet"
VALD_PATH  = PROC/"val_deploy.parquet"
TEST_PATH  = PROC/"test.parquet"
TDEP_PATH  = PROC/"test_deploy.parquet"

# Output folders and files
REPORTS = ROOT/"reports"; REPORTS.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPORTS/"csv";  CSV_DIR.mkdir(parents=True, exist_ok=True)
MODELS  = ROOT/"models"; MODELS.mkdir(parents=True, exist_ok=True)
MODEL_OUT = MODELS/f"{TAG}.joblib"
REPORT_JSON = REPORTS/f"{TAG}_report.json"

# Backbone model sources
EMO_MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"
BERT_LOCAL_CANDS = [ROOT/"models"/"bert_preproc256_v1", ROOT/"artifacts"/"bert_preproc256_v1"]
BERT_DIR = next((p for p in BERT_LOCAL_CANDS if p.exists()), None)
if BERT_DIR is None:
    raise SystemExit("Best BERT not found. Expected models/bert_preproc256_v1 or artifacts/bert_preproc256_v1.")

# Batch sizes and tokenization lengths
BATCH_SIZE = 64
EMO_MAX_LEN = 128
BERT_MAX_LEN = 256

# Set seeds for Python, NumPy, and PyTorch
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Pick device in order: CUDA, then Apple MPS, then CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

# Import normalize_text from shared scripts with fallbacks
sys.path += [str(ROOT), str(ROOT/"scripts")]
try:
    from scripts.text_preprocess import normalize_text  # type: ignore
except Exception:
    try:
        from text_preprocess import normalize_text  # type: ignore
    except Exception:
        def normalize_text(s: str) -> str:
            return str(s).strip()

# Minimal dataset for DataLoader
class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

# Get per-emotion probabilities from the GoEmotions model
@torch.no_grad()
def emo_probs(texts: List[str], tok, model) -> np.ndarray:
    dl = DataLoader(TextDataset(texts), batch_size=BATCH_SIZE, shuffle=False)
    outs = []
    model.eval()
    for batch in dl:
        enc = tok(list(batch), truncation=True, padding=True, max_length=EMO_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        outs.append(probs)
    return np.concatenate(outs, axis=0)

# Get the positive class probability from the local BERT smishing classifier
@torch.no_grad()
def bert_pos_prob(texts: List[str], tok, model) -> np.ndarray:
    dl = DataLoader(TextDataset(texts), batch_size=BATCH_SIZE, shuffle=False)
    outs = []
    model.eval()
    for batch in dl:
        enc = tok(list(batch), truncation=True, padding=True, max_length=BERT_MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        outs.append(probs)
    return np.concatenate(outs, axis=0)

# Choose one threshold that maximizes F1 on validation
def f1_optimal_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    thr = np.r_[0.0, thr]  # align thresholds with precision/recall arrays
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(f1s))])

# Compute standard metrics at a fixed threshold
def eval_at_threshold(y_true, y_prob, thr=0.5) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "threshold": float(thr), "accuracy": float(acc), "precision": float(prec),
        "recall": float(rec), "f1": float(f1), "roc_auc": float(roc), "pr_auc": float(ap),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

# Save probabilities and labels for downstream evaluation scripts
def save_preds(tag: str, split: str, y: np.ndarray, p: np.ndarray):
    df = pd.DataFrame({"label": y.astype(int), "prob_smish": p.astype(float)})
    df.to_csv(REPORTS / f"predictions_{tag}_{split}_with_probs.csv", index=False)

# Main fusion pipeline
def main():
    # Helper to load one split and enforce schema and normalization
    def load_df(p: Path) -> pd.DataFrame:
        if not p.exists():
            raise SystemExit(f"Missing {p}.")
        df = pd.read_parquet(p).dropna(subset=["text","label"]).copy()
        df["text"] = df["text"].astype(str).map(normalize_text)
        df["label"] = df["label"].astype(int)
        return df

    # Load all protocol splits
    train = load_df(TRAIN_PATH)
    vdep  = load_df(VALD_PATH)
    test  = load_df(TEST_PATH)
    tdep  = load_df(TDEP_PATH)

    # Load emotion backbone (frozen)
    emo_tok = AutoTokenizer.from_pretrained(EMO_MODEL_NAME)
    emo_mod = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL_NAME).to(DEVICE)

    # Load best BERT backbone from local folder (frozen)
    bert_tok = AutoTokenizer.from_pretrained(BERT_DIR)
    bert_mod = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(DEVICE)

    # Build features per split: concatenate emotion probs with BERT positive probability
    def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        e = emo_probs(df["text"].tolist(), emo_tok, emo_mod)
        b = bert_pos_prob(df["text"].tolist(), bert_tok, bert_mod)[:, None]
        X = np.concatenate([e, b], axis=1)
        y = df["label"].values
        return X, y

    # Compute features for all splits
    print("[feat] train…")
    Xtr, ytr = build_features(train)
    print("[feat] val_deploy…")
    Xvd, yvd = build_features(vdep)
    print("[feat] test…")
    Xte, yte = build_features(test)
    print("[feat] test_deploy…")
    Xtd, ytd = build_features(tdep)

    # Fit a balanced Logistic Regression on the fused features
    classes = np.unique(ytr)
    class_weight = None
    if len(classes) == 2:
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=ytr)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight, solver='lbfgs')
    clf.fit(Xtr, ytr)

    # Pick a single F1-max threshold on val_deploy
    p_vd = clf.predict_proba(Xvd)[:, 1]
    thr  = f1_optimal_threshold(yvd, p_vd)

    # Evaluate on test and test_deploy with that fixed threshold
    p_te = clf.predict_proba(Xte)[:, 1]
    p_td = clf.predict_proba(Xtd)[:, 1]

    res_test = eval_at_threshold(yte, p_te, thr)
    res_tdep = eval_at_threshold(ytd, p_td, thr)

    # Save model artifact with small metadata
    joblib.dump({
        "model": clf,
        "emo_model": EMO_MODEL_NAME,
        "bert_dir": str(BERT_DIR),
        "feature_dims": {"emotions": Xtr.shape[1]-1, "bert_prob": 1},
        "seed": SEED
    }, MODEL_OUT)

    # Save a JSON report with settings, threshold, and metrics
    with open(REPORT_JSON, "w") as f:
        json.dump({
            "model": TAG,
            "backbones": {"emotions": EMO_MODEL_NAME, "bert": str(BERT_DIR)},
            "seed": SEED,
            "threshold": float(thr),
            "metrics": {"test": res_test, "test_deploy": res_tdep}
        }, f, indent=2)

    # Write probability CSVs for later plotting and comparison
    save_preds(TAG, "val_deploy", yvd, p_vd)
    save_preds(TAG, "test",       yte, p_te)
    save_preds(TAG, "test_deploy",ytd, p_td)

    # Small helper to append one row to a CSV file
    def write_csv_row(path: Path, row: dict, header=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        import csv as _csv
        if header is None: header = list(row.keys())
        with path.open("a", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=header)
            if write_header: w.writeheader()
            w.writerow(row)

    # Log training distribution for auditing
    train_row = {
        "run": TAG, "train_split":"train_balanced",
        "total": int(len(train)),
        "ham": int((train.label==0).sum()),
        "smish": int((train.label==1).sum()),
        "smish_pct": round(train.label.mean()*100.0, 3),
        "ngram_max":"", "min_df":"", "max_df":"", "C":"", "alpha":"",
        "seed": SEED, "class_weight":"balanced"
    }
    write_csv_row(CSV_DIR/"baseline_train_log.csv", train_row,
                  header=["run","train_split","total","ham","smish","smish_pct","ngram_max","min_df","max_df","C","alpha","seed","class_weight"])

    # Log metrics for test and test_deploy
    for split_name, res in [("test", res_test), ("test_deploy", res_tdep)]:
        metrics_row = {
            "split": split_name, "model": TAG, "threshold": res["threshold"],
            "accuracy": res["accuracy"], "precision": res["precision"], "recall": res["recall"],
            "f1": res["f1"], "roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"],
            "tn": res["tn"], "fp": res["fp"], "fn": res["fn"], "tp": res["tp"],
            "ngram_max":"", "min_df":"", "max_df":"", "C":"", "alpha":"", "seed": SEED, "train_split":"train_balanced"
        }
        write_csv_row(CSV_DIR/"baseline_metrics_log.csv", metrics_row,
                      header=["split","model","threshold","accuracy","precision","recall","f1","roc_auc","pr_auc","tn","fp","fn","tp","ngram_max","min_df","max_df","C","alpha","seed","train_split"])

    # Short status so expected files are easy to confirm
    print("\n==== emotion_fusion_model_1 complete ====")
    print(f"Device: {DEVICE}")
    print(f"Best BERT dir: {BERT_DIR}")
    print(f"Threshold (val_deploy, F1-max): {thr:.6f}")
    print(f"Saved model:  {MODEL_OUT}")
    print(f"Saved report: {REPORT_JSON}")
    print("Pred CSVs → reports/predictions_emotion_fusion_model_1_{val_deploy,test,test_deploy}_with_probs.csv")

if __name__ == "__main__":
    main()
