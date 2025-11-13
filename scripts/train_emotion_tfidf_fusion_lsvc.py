#!/usr/bin/env python3
# scripts/run_emotion_fusion_tfidf_lsvc
#.py

# Late fusion: combine GoEmotions probabilities and a TF-IDF LinearSVC smish score with Logistic Regression.

# Protocol
# Train features on train_balanced.
# Pick one threshold on val_deploy that maximizes F1.
# Evaluate on test and test_deploy with that fixed threshold.
#
# Backbones (frozen during fusion)
# Emotions: joeddav/distilbert-base-uncased-go-emotions-student (softmax to per-emotion probabilities).
# TF-IDF: TF-IDF(ngram<=2, min_df=2, max_df=0.95) + LinearSVC calibrated to probabilities with sigmoid.
#
# INPUTS
# data/processed/train_balanced.parquet
# data/processed/val_deploy.parquet
# data/processed/test.parquet
# data/processed/test_deploy.parquet
#
# OUTPUTS
# models/emotion_fusion_tfidf_lsvc_1.joblib
# reports/emotion_fusion_tfidf_lsvc_1_report.json
# reports/csv/baseline_train_log.csv
# reports/csv/baseline_metrics_log.csv
# reports/predictions_emotion_fusion_tfidf_lsvc_1_{val_deploy,test,test_deploy}_with_probs.csv

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json, sys, random

import numpy as np
import pandas as pd
import joblib

# Torch and Transformers for GoEmotions backbone
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# TF-IDF LinearSVC backbone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Meta model and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

# Fixed config so runs are reproducible
SEED = 1337
TAG  = "emotion_fusion_tfidf_lsvc_1"

# Data paths under repo root
ROOT = Path(".")
PROC = ROOT/"data"/"processed"
TRAIN_PATH = PROC/"train_balanced.parquet"
VALD_PATH  = PROC/"val_deploy.parquet"
TEST_PATH  = PROC/"test.parquet"
TDEP_PATH  = PROC/"test_deploy.parquet"

# Output folders for reports and models
REPORTS = ROOT/"reports"; REPORTS.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPORTS/"csv";  CSV_DIR.mkdir(parents=True, exist_ok=True)
MODELS  = ROOT/"models"; MODELS.mkdir(parents=True, exist_ok=True)
MODEL_OUT = MODELS/f"{TAG}.joblib"
REPORT_JSON = REPORTS/f"{TAG}_report.json"

# GoEmotions model name on Hugging Face
EMO_MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

# TF-IDF hyperparameters (match baseline settings)
NGRAM_MAX = 2
MIN_DF    = 2
MAX_DF    = 0.95

# Batch size and max length for emotion model
BATCH_SIZE  = 64
EMO_MAX_LEN = 128

# Set seeds for repeatability
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Pick device: CUDA then MPS then CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

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

# Simple dataset wrapper for plain text lists
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

# Choose probability threshold that maximizes F1 on validation
def f1_optimal_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    thr = np.r_[0.0, thr]  # align thresholds with precision/recall
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(f1s))])

# Compute metrics at a given threshold
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

# Write prediction CSVs with labels and probabilities
def save_preds(tag: str, split: str, y: np.ndarray, p: np.ndarray):
    df = pd.DataFrame({"label": y.astype(int), "prob_smish": p.astype(float)})
    df.to_csv(REPORTS / f"predictions_{tag}_{split}_with_probs.csv", index=False)

# Append one row to a CSV log file
def write_csv_row(path: Path, row: dict, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if header is None: header = list(row.keys())
    with path.open("a", newline="") as fh:
        import csv as _csv
        w = _csv.DictWriter(fh, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row)

# Main training and evaluation flow
def main():
    # Load and clean each split
    def load_df(p: Path) -> pd.DataFrame:
        if not p.exists():
            raise SystemExit(f"Missing {p}.")
        df = pd.read_parquet(p).dropna(subset=["text","label"]).copy()
        df["text"] = df["text"].astype(str).map(normalize_text)
        df["label"] = df["label"].astype(int)
        return df

    train = load_df(TRAIN_PATH)
    vdep  = load_df(VALD_PATH)
    test  = load_df(TEST_PATH)
    tdep  = load_df(TDEP_PATH)

    # Build the emotions backbone
    emo_tok = AutoTokenizer.from_pretrained(EMO_MODEL_NAME)
    emo_mod = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL_NAME).to(DEVICE)

    # Fit TF-IDF + LinearSVC on train_balanced and calibrate to probabilities
    vect = TfidfVectorizer(ngram_range=(1, NGRAM_MAX), min_df=MIN_DF, max_df=MAX_DF)
    Xtr_tfidf = vect.fit_transform(train["text"].tolist())
    lsvc = LinearSVC(random_state=SEED)
    clf_cal = CalibratedClassifierCV(lsvc, method="sigmoid", cv=5)
    clf_cal.fit(Xtr_tfidf, train["label"].values)

    # Compute calibrated TF-IDF probabilities for any text list
    def tfidf_pos_prob(texts: List[str]) -> np.ndarray:
        X = vect.transform(texts)
        p = clf_cal.predict_proba(X)[:, 1]
        return p

    # Build fusion features: all emotion probs plus the TF-IDF smish probability
    def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        e = emo_probs(df["text"].tolist(), emo_tok, emo_mod)
        b = tfidf_pos_prob(df["text"].tolist())[:, None]
        X = np.concatenate([e, b], axis=1)
        y = df["label"].values
        return X, y

    # Extract features for each split
    print("[feat] train…")
    Xtr, ytr = build_features(train)
    print("[feat] val_deploy…")
    Xvd, yvd = build_features(vdep)
    print("[feat] test…")
    Xte, yte = build_features(test)
    print("[feat] test_deploy…")
    Xtd, ytd = build_features(tdep)

    # Fit the meta Logistic Regression with class weights if needed
    classes = np.unique(ytr)
    class_weight = None
    if len(classes) == 2:
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=ytr)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight, solver='lbfgs')
    meta.fit(Xtr, ytr)

    # Pick the F1-max threshold on val_deploy
    p_vd = meta.predict_proba(Xvd)[:, 1]
    thr  = f1_optimal_threshold(yvd, p_vd)

    # Evaluate on test and test_deploy with the fixed threshold
    p_te = meta.predict_proba(Xte)[:, 1]
    p_td = meta.predict_proba(Xtd)[:, 1]

    res_test = eval_at_threshold(yte, p_te, thr)
    res_tdep = eval_at_threshold(ytd, p_td, thr)

    # Save the fusion model and components needed for reuse
    joblib.dump({
        "fusion_model": meta,
        "emo_model": EMO_MODEL_NAME,
        "tfidf": {
            "vectorizer": vect,
            "calibrated_clf": clf_cal,
            "params": {"ngram_max": NGRAM_MAX, "min_df": MIN_DF, "max_df": MAX_DF}
        },
        "feature_dims": {"emotions": Xtr.shape[1]-1, "tfidf_prob": 1},
        "seed": SEED
    }, MODEL_OUT)

    # Write a JSON report with settings, threshold, and metrics
    with open(REPORT_JSON, "w") as f:
        json.dump({
            "model": TAG,
            "backbones": {
                "emotions": EMO_MODEL_NAME,
                "tfidf": "LinearSVC + CalibratedClassifierCV(sigmoid)"
            },
            "seed": SEED,
            "threshold": float(thr),
            "metrics": {"test": res_test, "test_deploy": res_tdep}
        }, f, indent=2)

    # Save prediction CSVs for evaluator and plots
    save_preds(TAG, "val_deploy", yvd, p_vd)
    save_preds(TAG, "test",       yte, p_te)
    save_preds(TAG, "test_deploy",ytd, p_td)

    # Append a training summary row to the shared CSV log
    train_row = {
        "run": TAG, "train_split":"train_balanced",
        "total": int(len(train)),
        "ham": int((train.label==0).sum()),
        "smish": int((train.label==1).sum()),
        "smish_pct": round(train.label.mean()*100.0, 3),
        "ngram_max": NGRAM_MAX, "min_df": MIN_DF, "max_df": MAX_DF,
        "C":"", "alpha":"", "seed": SEED, "class_weight":"balanced"
    }
    write_csv_row(
        CSV_DIR/"baseline_train_log.csv",
        train_row,
        header=["run","train_split","total","ham","smish","smish_pct","ngram_max","min_df","max_df","C","alpha","seed","class_weight"]
    )

    # Append metrics rows for test and test_deploy
    for split_name, res in [("test", res_test), ("test_deploy", res_tdep)]:
        metrics_row = {
            "split": split_name, "model": TAG, "threshold": res["threshold"],
            "accuracy": res["accuracy"], "precision": res["precision"], "recall": res["recall"],
            "f1": res["f1"], "roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"],
            "tn": res["tn"], "fp": res["fp"], "fn": res["fn"], "tp": res["tp"],
            "ngram_max": NGRAM_MAX, "min_df": MIN_DF, "max_df": MAX_DF,
            "C":"", "alpha":"", "seed": SEED, "train_split":"train_balanced"
        }
        write_csv_row(
            CSV_DIR/"baseline_metrics_log.csv",
            metrics_row,
            header=["split","model","threshold","accuracy","precision","recall","f1","roc_auc","pr_auc","tn","fp","fn","tp","ngram_max","min_df","max_df","C","alpha","seed","train_split"]
        )

    # Print short status so outputs are easy to confirm
    print("\n==== emotion_fusion_tfidf_lsvc_1 complete ====")
    print(f"Device: {DEVICE}")
    print(f"Threshold (val_deploy, F1-max): {thr:.6f}")
    print(f"Saved model:  {MODEL_OUT}")
    print(f"Saved report: {REPORT_JSON}")
    print("Pred CSVs → reports/predictions_emotion_fusion_tfidf_lsvc_1_{val_deploy,test,test_deploy}_with_probs.csv")

if __name__ == "__main__":
    main()
