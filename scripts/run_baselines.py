#!/usr/bin/env python3
"""
Train AND evaluate all classical baselines (no CLI switches).

Models:
  - tfidf_lr        : TF-IDF + LogisticRegression (class_weight="balanced")
  - tfidf_linearsvc : TF-IDF + LinearSVC (class_weight="balanced")
  - tfidf_mnb       : TF-IDF + MultinomialNB

Training split:  data/processed/train.parquet
Evaluation:      data/processed/test.parquet, data/processed/test_deploy.parquet
Threshold:       0.5 (LinearSVC uses decision_function mapped via sigmoid)

Outputs (overwrite each run):
  models/tfidf_lr.joblib
  models/tfidf_linearsvc.joblib
  models/tfidf_mnb.joblib
  reports/tfidf_lr_report.json
  reports/tfidf_linearsvc_report.json
  reports/tfidf_mnb_report.json
Appends rows:
  reports/csv/baseline_train_log.csv
  reports/csv/baseline_metrics_log.csv
"""

from pathlib import Path
import json, csv
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix
)
from scipy.special import expit

# -------- fixed hyperparams (edit here if needed) --------
SEED = 1337
NGRAM_MAX = 2
MIN_DF = 2
MAX_DF = 0.95
C = 1.0
ALPHA = 1.0
THRESHOLD = 0.5

# ---------------- paths ----------------
PROC    = Path("data/processed")
MODELS  = Path("models");  MODELS.mkdir(parents=True, exist_ok=True)
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPORTS / "csv"; CSV_DIR.mkdir(parents=True, exist_ok=True)

# ------------- io helpers -------------
def write_csv_row(path: Path, row: dict, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if header is None: header = list(row.keys())
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row)

def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2))

# ------------- data utils -------------
def load_split(name: str) -> pd.DataFrame:
    p = PROC / f"{name}.parquet"
    if not p.exists(): raise SystemExit(f"Missing {p}.")
    df = pd.read_parquet(p)
    if not {"text","label"}.issubset(df.columns):
        raise SystemExit(f"{p} must have columns: text, label")
    return df.dropna(subset=["text","label"]).assign(text=lambda d: d["text"].astype(str))

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, NGRAM_MAX),
        min_df=MIN_DF,
        max_df=MAX_DF,
        lowercase=False
    )

# ---------- evaluation helper ----------
def eval_with_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else float("nan")
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true))==2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return dict(threshold=float(thr), accuracy=float(acc), precision=float(prec), recall=float(rec),
                f1=float(f1), roc_auc=float(roc), pr_auc=float(ap),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

# --------------- runners ---------------
def run_tfidf_lr(train_df):
    vect = build_vectorizer()
    X = vect.fit_transform(train_df["text"])
    y = train_df["label"].astype(int).values
    clf = LogisticRegression(
        C=C, class_weight="balanced",
        max_iter=2000, solver="liblinear", random_state=SEED
    )
    clf.fit(X, y)
    return vect, clf

def run_tfidf_linearsvc(train_df):
    vect = build_vectorizer()
    X = vect.fit_transform(train_df["text"])
    y = train_df["label"].astype(int).values
    clf = LinearSVC(C=C, class_weight="balanced", random_state=SEED)
    clf.fit(X, y)
    return vect, clf

def run_tfidf_mnb(train_df):
    vect = build_vectorizer()
    X = vect.fit_transform(train_df["text"])
    y = train_df["label"].astype(int).values
    clf = MultinomialNB(alpha=ALPHA)
    clf.fit(X, y)
    return vect, clf

# ---------------- main -----------------
def main():
    # fixed splits
    train_df = load_split("train")
    test_df  = load_split("test")
    testd_df = load_split("test_deploy")

    # shared training summary row template
    train_row_base = {
        "train_split":"train",
        "total": int(len(train_df)),
        "ham":   int((train_df.label==0).sum()),
        "smish": int((train_df.label==1).sum()),
        "smish_pct": round(train_df.label.mean()*100.0, 3),
        "ngram_max": NGRAM_MAX, "min_df": MIN_DF, "max_df": MAX_DF,
        "C": C, "alpha": ALPHA, "seed": SEED, "class_weight": "balanced"
    }

    # ===== 1) TF-IDF + LogisticRegression =====
    print("\n[tfidf_lr] training & evaluating…")
    vect, clf = run_tfidf_lr(train_df)
    dump({"vectorizer": vect, "model": clf,
          "params": {"C":C,"ngram_max":NGRAM_MAX,"min_df":MIN_DF,"max_df":MAX_DF,"seed":SEED},
          "train_split": "train"}, MODELS / "tfidf_lr.joblib")

    X_te = vect.transform(test_df["text"]);     y_te = test_df["label"].astype(int).values
    X_td = vect.transform(testd_df["text"]);    y_td = testd_df["label"].astype(int).values
    p_te = clf.predict_proba(X_te)[:,1] if hasattr(clf,"predict_proba") else expit(clf.decision_function(X_te))
    p_td = clf.predict_proba(X_td)[:,1] if hasattr(clf,"predict_proba") else expit(clf.decision_function(X_td))
    res_test = eval_with_threshold(y_te, p_te, THRESHOLD)
    res_td   = eval_with_threshold(y_td, p_td, THRESHOLD)

    write_csv_row(CSV_DIR/"baseline_train_log.csv",
                  {"run":"tfidf_lr", **train_row_base},
                  header=["run","train_split","total","ham","smish","smish_pct","ngram_max","min_df","max_df","C","alpha","seed","class_weight"])
    for split_name, res in [("test", res_test), ("test_deploy", res_td)]:
        write_csv_row(CSV_DIR/"baseline_metrics_log.csv",
            {"split":split_name,"model":"tfidf_lr","threshold":res["threshold"],
             "accuracy":res["accuracy"],"precision":res["precision"],"recall":res["recall"],
             "f1":res["f1"],"roc_auc":res["roc_auc"],"pr_auc":res["pr_auc"],
             "tn":res["tn"],"fp":res["fp"],"fn":res["fn"],"tp":res["tp"],
             "ngram_max":NGRAM_MAX,"min_df":MIN_DF,"max_df":MAX_DF,"C":C,"alpha":"","seed":SEED,"train_split":"train"},
            header=["split","model","threshold","accuracy","precision","recall","f1","roc_auc","pr_auc","tn","fp","fn","tp","ngram_max","min_df","max_df","C","alpha","seed","train_split"])
    save_json({"model":"tfidf_lr","params":{"C":C,"ngram_max":NGRAM_MAX,"min_df":MIN_DF,"max_df":MAX_DF,"seed":SEED},
               "train_split":"train","threshold":THRESHOLD,"metrics":{"test":res_test,"test_deploy":res_td}},
              REPORTS/"tfidf_lr_report.json")
    print("[tfidf_lr] done.")

    # ===== 2) TF-IDF + LinearSVC =====
    print("\n[tfidf_linearsvc] training & evaluating…")
    vect, clf = run_tfidf_linearsvc(train_df)
    dump({"vectorizer": vect, "model": clf,
          "params": {"C":C,"ngram_max":NGRAM_MAX,"min_df":MIN_DF,"max_df":MAX_DF,"seed":SEED},
          "train_split": "train"}, MODELS / "tfidf_linearsvc.joblib")

    X_te = vect.transform(test_df["text"]); X_td = vect.transform(testd_df["text"])
    p_te, p_td = expit(clf.decision_function(X_te)), expit(clf.decision_function(X_td))
    res_test = eval_with_threshold(y_te, p_te, THRESHOLD)
    res_td   = eval_with_threshold(y_td, p_td, THRESHOLD)

    write_csv_row(CSV_DIR/"baseline_train_log.csv",
                  {"run":"tfidf_linearsvc", **train_row_base})
    for split_name, res in [("test", res_test), ("test_deploy", res_td)]:
        write_csv_row(CSV_DIR/"baseline_metrics_log.csv",
            {"split":split_name,"model":"tfidf_linearsvc","threshold":res["threshold"],
             "accuracy":res["accuracy"],"precision":res["precision"],"recall":res["recall"],
             "f1":res["f1"],"roc_auc":res["roc_auc"],"pr_auc":res["pr_auc"],
             "tn":res["tn"],"fp":res["fp"],"fn":res["fn"],"tp":res["tp"],
             "ngram_max":NGRAM_MAX,"min_df":MIN_DF,"max_df":MAX_DF,"C":C,"alpha":"","seed":SEED,"train_split":"train"})
    save_json({"model":"tfidf_linearsvc","params":{"C":C,"ngram_max":NGRAM_MAX,"min_df":MIN_DF,"max_df":MAX_DF,"seed":SEED},
               "train_split":"train","threshold":THRESHOLD,"metrics":{"test":res_test,"test_deploy":res_td}},
              REPORTS/"tfidf_linearsvc_report.json")
    print("[tfidf_linearsvc] done.")

    # ===== 3) TF-IDF + MultinomialNB =====
    print("\n[tfidf_mnb] training & evaluating…")
    vect, clf = run_tfidf_mnb(train_df)
    dump({"vectorizer": vect, "model": clf,
          "params": {"alpha":ALPHA,"ngram_max":NGRAM_MAX,"min_df":MIN_DF,"max_df":MAX_DF,"seed":SEED},
          "train_split": "train"}, MODELS / "tfidf_mnb.joblib")

    X_te = vect.transform(test_df["text"]); X_td = vect.transform(testd_df["text"])
    p_te = clf.predict_proba(X_te)[:,1] if hasattr(clf,"predict_proba") else expit(clf.decision_function(X_te))
    p_td = clf.predict_proba(X_td)[:,1] if hasattr(clf,"predict_proba") else expit(clf.decision_function(X_td))
    res_test = eval_with_threshold(y_te, p_te, THRESHOLD)
    res_td   = eval_with_threshold(y_td, p_td, THRESHOLD)

    write_csv_row(CSV_DIR/"baseline_train_log.csv",
                  {"run":"tfidf_mnb", **{**train_row_base, "class_weight":"none"}})
    for split_name, res in [("test", res_test), ("test_deploy", res_td)]:
        write_csv_row(CSV_DIR/"baseline_metrics_log.csv",
            {"split":split_name,"model":"tfidf_mnb","threshold":res["threshold"],
             "accuracy":res["accuracy"],"precision":res["precision"],"recall":res["recall"],
             "f1":res["f1"],"roc_auc":res["roc_auc"],"pr_auc":res["pr_auc"],
             "tn":res["tn"],"fp":res["fp"],"fn":res["fn"],"tp":res["tp"],
             "ngram_max":NGRAM_MAX,"min_df":MIN_DF,"max_df":MAX_DF,"C":"","alpha":ALPHA,"seed":SEED,"train_split":"train"})
    save_json({"model":"tfidf_mnb","params":{"alpha":ALPHA,"ngram_max":NGRAM_MAX,"min_df":MIN_DF,"max_df":MAX_DF,"seed":SEED},
               "train_split":"train","threshold":THRESHOLD,"metrics":{"test":res_test,"test_deploy":res_td}},
              REPORTS/"tfidf_mnb_report.json")
    print("[tfidf_mnb] done.")

    print("\nAll baselines trained & evaluated.")
    print("Models → models/")
    print("Logs   → reports/csv/baseline_train_log.csv, reports/csv/baseline_metrics_log.csv")
    print("JSON   → reports/tfidf_*_report.json")

if __name__ == "__main__":
    main()
