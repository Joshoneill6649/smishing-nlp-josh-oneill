#!/usr/bin/env python3
# Josh O’Neill • x23315369 — Baselines on a chosen train file (TF-IDF + LR/SVM/NB)

import json, pathlib, joblib, argparse
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_recall_curve, classification_report,
                             confusion_matrix, roc_auc_score, average_precision_score,
                             roc_curve)
import matplotlib.pyplot as plt

SEED = 23315369
ROOT = pathlib.Path(".")
PRO  = ROOT/"data/processed"
REP  = ROOT/"reports"
FIG  = REP/"figures"
MOD  = ROOT/"models"
for p in [REP, FIG, MOD]: p.mkdir(parents=True, exist_ok=True)

def ybin(lbl): return (lbl=="smish").astype(int).to_numpy()

def pick_thr_by_f1(y_true, p_smish):
    P,R,T = precision_recall_curve(y_true, p_smish)
    F1 = (2*P*R)/(P+R+1e-9)
    i = int(np.nanargmax(F1))
    thr = 0.5 if i==len(T) else float(T[i])
    return thr, float(F1[i]), float(P[i]), float(R[i])

def eval_split(tag, split_name, y_true, p_smish, thr, results_rows, results_tag):
    y_pred = (p_smish >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=["ham","smish"], output_dict=True)
    cm  = confusion_matrix(y_true, y_pred).tolist()
    roc = roc_auc_score(y_true, p_smish)
    pr  = average_precision_score(y_true, p_smish)

    fpr, tpr, _ = roc_curve(y_true, p_smish)
    plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {tag} ({split_name}) [{results_tag}]")
    plt.tight_layout(); plt.savefig(FIG/f"{tag}_{results_tag}_{split_name}_roc.png", dpi=140); plt.close()

    P,R,_ = precision_recall_curve(y_true, p_smish)
    plt.figure(); plt.plot(R,P); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {tag} ({split_name}) [{results_tag}]")
    plt.tight_layout(); plt.savefig(FIG/f"{tag}_{results_tag}_{split_name}_pr.png", dpi=140); plt.close()

    results_rows.append({
        "run_tag": results_tag,
        "model": tag,
        "split": split_name,
        "threshold": thr,
        "accuracy": acc,
        "precision_smish": rep["smish"]["precision"],
        "recall_smish":    rep["smish"]["recall"],
        "f1_smish":        rep["smish"]["f1-score"],
        "f1_macro":        rep["macro avg"]["f1-score"],
        "roc_auc": roc,
        "pr_auc":  pr,
        "cm": json.dumps(cm)
    })

def run_model(tag, model, train_df, val_df, test_df, results_path, results_tag):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.98)
    pipe = make_pipeline(vec, model)

    Xtr, ytr = train_df["text_clean"], ybin(train_df["label"])
    Xva, yva = val_df["text_clean"],   ybin(val_df["label"])
    Xte, yte = test_df["text_clean"],  ybin(test_df["label"])

    pipe.fit(Xtr, ytr)
    joblib.dump(pipe, MOD/f"{tag}_{results_tag}.joblib")

    def proba(p, X):
        if hasattr(p, "predict_proba"): return p.predict_proba(X)[:,1]
        if hasattr(p, "decision_function"):
            s = p.decision_function(X)
            return (s - s.min()) / (s.max() - s.min() + 1e-9)
        return p.predict(X).astype(float)

    rows = []
    p_val = proba(pipe, Xva)
    thr, f1, p_at, r_at = pick_thr_by_f1(yva, p_val)
    eval_split(tag, "val",  yva, p_val, thr, rows, results_tag)

    p_test = proba(pipe, Xte)
    eval_split(tag, "test", yte, p_test, thr, rows, results_tag)

    df_res = pd.DataFrame(rows)
    if results_path.exists():
        prev = pd.read_csv(results_path)
        df_res = pd.concat([prev, df_res], ignore_index=True)
    df_res.to_csv(results_path, index=False)
    print(f"[{tag}::{results_tag}] thr={thr:.3f} | val F1={f1:.3f} (P={p_at:.3f}, R={r_at:.3f}) -> {results_path}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", type=str, default=str(PRO/"train_balanced.parquet"),
                    help="Parquet file for training (e.g., data/processed/train_balanced_equal.parquet)")
    ap.add_argument("--results", type=str, default=str(REP/"baseline_results.csv"),
                    help="CSV to write/append results to")
    ap.add_argument("--tag", type=str, default="default",
                    help="Run tag appended to filenames and results rows")
    args = ap.parse_args()

    train_df = pd.read_parquet(args.train_file)
    val_df   = pd.read_parquet(PRO/"val.parquet")
    test_df  = pd.read_parquet(PRO/"test.parquet")

    results_path = pathlib.Path(args.results)
    tag = args.tag

    # Models
    lr  = LogisticRegression(max_iter=5000, n_jobs=-1, class_weight="balanced")
    svm = CalibratedClassifierCV(estimator=LinearSVC(class_weight="balanced", random_state=SEED),  # <- FIXED
                                 cv=3, method="sigmoid")
    nb  = MultinomialNB()

    run_model("baseline_logreg", lr,  train_df, val_df, test_df, results_path, tag)
    run_model("baseline_svm",   svm, train_df, val_df, test_df, results_path, tag)
    run_model("baseline_mnb",   nb,  train_df, val_df, test_df, results_path, tag)
