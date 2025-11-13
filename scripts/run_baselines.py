#!/usr/bin/env python3
# scripts/run_baselines.py
# Train and evaluate 3 classical baselines using the same protocol
# TF IDF with LogisticRegression, TF IDF with LinearSVC, TF IDF with MultinomialNB
# Threshold is chosen on val_deploy by F1 max and then used on test and test_deploy
#
# INPUTS
# data/processed/train_balanced.parquet
# data/processed/val_deploy.parquet
# data/processed/test.parquet
# data/processed/test_deploy.parquet
#
# OUTPUTS
# models/tfidf_lr.joblib
# models/tfidf_linearsvc.joblib
# models/tfidf_mnb.joblib
# reports/tfidf_lr_report.json
# reports/tfidf_linearsvc_report.json
# reports/tfidf_mnb_report.json
# reports/predictions_<tag>_{val_deploy,test,test_deploy}_with_probs.csv
# reports/csv/baseline_train_log.csv    appends a row per model
# reports/csv/baseline_metrics_log.csv  appends a row per model per split

from pathlib import Path
import json
import csv
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve
)
from scipy.special import expit

# FIXED HYPERPARAMS
SEED = 1337
NGRAM_MAX = 2
MIN_DF = 2
MAX_DF = 0.95
C = 1.0
ALPHA = 1.0

# PATHS
PROC    = Path("data/processed")
MODELS  = Path("models");  MODELS.mkdir(parents=True, exist_ok=True)
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPORTS / "csv"; CSV_DIR.mkdir(parents=True, exist_ok=True)

# IO HELPERS
def write_csv_row(path: Path, row: dict, header=None):
    # Append a single row to a CSV and write a header once
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if header is None:
        header = list(row.keys())
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

def save_json(obj: dict, out_path: Path):
    # Write a JSON with pretty indent
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2))

def save_preds(tag: str, split: str, y: np.ndarray, p: np.ndarray):
    # Save labels and positive class probabilities for later analysis
    df = pd.DataFrame({"label": y.astype(int), "prob_smish": p.astype(float)})
    df.to_csv(REPORTS / f"predictions_{tag}_{split}_with_probs.csv", index=False)

# DATA UTILS
def load_split(name: str) -> pd.DataFrame:
    # Read a named parquet split and coerce to expected dtypes
    p = PROC / f"{name}.parquet"
    if not p.exists():
        raise SystemExit(f"Missing {p}.")
    df = pd.read_parquet(p)
    if not {"text", "label"}.issubset(df.columns):
        raise SystemExit(f"{p} must have columns: text, label")
    return df.dropna(subset=["text", "label"]).assign(
        text=lambda d: d["text"].astype(str),
        label=lambda d: d["label"].astype(int)
    )

def build_vectorizer() -> TfidfVectorizer:
    # TF IDF with 1 to NGRAM_MAX, document frequency bounds, keep case as is
    return TfidfVectorizer(
        ngram_range=(1, NGRAM_MAX),
        min_df=MIN_DF,
        max_df=MAX_DF,
        lowercase=False
    )

# THRESHOLD AND EVAL HELPERS
def pick_f1_threshold(y_true, y_prob):
    # Choose probability threshold that maximizes F1 on validation
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    thr = np.r_[0.0, thr]  # align with prec and rec arrays
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(thr[int(np.nanargmax(f1s))])

def eval_with_threshold(y_true, y_prob, thr: float):
    # Convert probabilities to labels at thr and compute metrics
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        threshold=float(thr), accuracy=float(acc), precision=float(prec), recall=float(rec),
        f1=float(f1), roc_auc=float(roc), pr_auc=float(ap),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )

def prob_fn(model, X):
    # Return positive class probabilities for any of the 3 models
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return expit(model.decision_function(X))  # LinearSVC margin to probability

# RUNNERS
def run_tfidf_lr(train_df):
    vect = build_vectorizer()
    X = vect.fit_transform(train_df["text"])
    y = train_df["label"].values
    clf = LogisticRegression(
        C=C, class_weight="balanced",
        max_iter=2000, solver="liblinear", random_state=SEED
    )
    clf.fit(X, y)
    return vect, clf

def run_tfidf_linearsvc(train_df):
    vect = build_vectorizer()
    X = vect.fit_transform(train_df["text"])
    y = train_df["label"].values
    clf = LinearSVC(C=C, class_weight="balanced", random_state=SEED)
    clf.fit(X, y)
    return vect, clf

def run_tfidf_mnb(train_df):
    vect = build_vectorizer()
    X = vect.fit_transform(train_df["text"])
    y = train_df["label"].values
    clf = MultinomialNB(alpha=ALPHA)
    clf.fit(X, y)
    return vect, clf

# MAIN
def main():
    # Load protocol splits
    train_df  = load_split("train_balanced")
    valdep_df = load_split("val_deploy")
    test_df   = load_split("test")
    testd_df  = load_split("test_deploy")

    # Shared training summary base
    train_row_base = {
        "train_split": "train_balanced",
        "total": int(len(train_df)),
        "ham":   int((train_df.label == 0).sum()),
        "smish": int((train_df.label == 1).sum()),
        "smish_pct": round(train_df.label.mean() * 100.0, 3),
        "ngram_max": NGRAM_MAX, "min_df": MIN_DF, "max_df": MAX_DF,
        "C": C, "alpha": ALPHA, "seed": SEED, "class_weight": "balanced"
    }

    def train_eval(tag, runner, mparams: dict):
        # Train model, pick threshold on val_deploy, evaluate on test and test_deploy, write artifacts
        print(f"\n[{tag}] training and evaluating")
        vect, clf = runner(train_df)
        dump({"vectorizer": vect, "model": clf,
              "params": mparams, "train_split": "train_balanced"}, MODELS / f"{tag}.joblib")

        # Threshold selection on val_deploy
        X_vd = vect.transform(valdep_df["text"]); y_vd = valdep_df["label"].values
        pv   = prob_fn(clf, X_vd)
        thr  = pick_f1_threshold(y_vd, pv)

        # Evaluate on test and test_deploy with the chosen threshold
        X_te = vect.transform(test_df["text"]);   y_te = test_df["label"].values
        X_td = vect.transform(testd_df["text"]);  y_td = testd_df["label"].values
        p_te = prob_fn(clf, X_te);  p_td = prob_fn(clf, X_td)

        res_test = eval_with_threshold(y_te, p_te, thr)
        res_td   = eval_with_threshold(y_td, p_td, thr)

        # Save prediction CSVs for later analysis
        save_preds(tag, "val_deploy", y_vd, pv)
        save_preds(tag, "test",       y_te, p_te)
        save_preds(tag, "test_deploy",y_td, p_td)

        # Append a training log row
        write_csv_row(
            CSV_DIR / "baseline_train_log.csv",
            {"run": tag, **train_row_base},
            header=["run","train_split","total","ham","smish","smish_pct",
                    "ngram_max","min_df","max_df","C","alpha","seed","class_weight"]
        )

        # Append metrics rows for test and test_deploy
        for split_name, res in [("test", res_test), ("test_deploy", res_td)]:
            write_csv_row(
                CSV_DIR / "baseline_metrics_log.csv",
                {
                    "split": split_name, "model": tag, "threshold": res["threshold"],
                    "accuracy": res["accuracy"], "precision": res["precision"], "recall": res["recall"],
                    "f1": res["f1"], "roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"],
                    "tn": res["tn"], "fp": res["fp"], "fn": res["fn"], "tp": res["tp"],
                    "ngram_max": NGRAM_MAX, "min_df": MIN_DF, "max_df": MAX_DF,
                    "C": mparams.get("C", ""), "alpha": mparams.get("alpha", ""),
                    "seed": SEED, "train_split": "train_balanced"
                },
                header=["split","model","threshold","accuracy","precision","recall","f1","roc_auc","pr_auc",
                        "tn","fp","fn","tp","ngram_max","min_df","max_df","C","alpha","seed","train_split"]
            )

        # Write per model report JSON
        save_json(
            {
                "model": tag, "params": mparams,
                "train_split": "train_balanced", "threshold": thr,
                "metrics": {"test": res_test, "test_deploy": res_td}
            },
            REPORTS / f"{tag}_report.json"
        )
        print(f"[{tag}] done. thr={thr:.4f}")

    # Run all 3 models
    train_eval(
        "tfidf_lr",
        run_tfidf_lr,
        {"C": C, "ngram_max": NGRAM_MAX, "min_df": MIN_DF, "max_df": MAX_DF, "seed": SEED}
    )
    train_eval(
        "tfidf_linearsvc",
        run_tfidf_linearsvc,
        {"C": C, "ngram_max": NGRAM_MAX, "min_df": MIN_DF, "max_df": MAX_DF, "seed": SEED}
    )
    train_eval(
        "tfidf_mnb",
        run_tfidf_mnb,
        {"alpha": ALPHA, "ngram_max": NGRAM_MAX, "min_df": MIN_DF, "max_df": MAX_DF, "seed": SEED}
    )

    print("\nAll baselines trained and evaluated")
    print("Models → models/")
    print("Logs   → reports/csv/baseline_train_log.csv, reports/csv/baseline_metrics_log.csv")
    print("JSON   → reports/tfidf_*_report.json")

if __name__ == "__main__":
    main()
