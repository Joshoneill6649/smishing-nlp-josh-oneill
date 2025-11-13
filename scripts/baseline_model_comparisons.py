#!/usr/bin/env python3
# baseline_model_comparisons.py
#
# Compare all trained models (baselines, transformers, fusion) and generate a leaderboard,
# full metrics tables, and standard figures for the deployment split.
# Sort models by PR-AUC then F1 then Recall on test_deploy.
#
# INPUTS 
# reports/*_report.json
# reports/predictions_<tag>_{test,test_deploy}_with_probs.csv
#
# OUTPUTS
# reports/csv/model_leaderboard.csv
# reports/csv/model_metrics_summary.csv
# reports/figures/model_compare/cm_<tag>_test_deploy.png
# reports/figures/model_compare/roc_<tag>_test_deploy.png
# reports/figures/model_compare/pr_<tag>_test_deploy.png
# reports/figures/model_compare/bars_test_deploy_f1_prauc.png

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

# Set root and output folders
ROOT = Path(".")
REPORTS = ROOT / "reports"
CSV_DIR = REPORTS / "csv"
FIG_DIR = REPORTS / "figures" / "model_compare"
CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Read a JSON file into a dict
def load_json(p: Path) -> dict:
    return json.loads(p.read_text())

# Try reading a predictions CSV for a given tag and split; normalize column names
def try_read_predictions(tag: str, split: str) -> pd.DataFrame | None:
    # Expected path pattern: reports/predictions_<tag>_<split>_with_probs.csv
    cand = REPORTS / f"predictions_{tag}_{split}_with_probs.csv"
    if not cand.exists():
        return None
    df = pd.read_csv(cand)
    # Standardize probability column if alternate names are present
    if "prob_smish" not in df.columns:
        for c in ["smish_prob", "p_smish", "prob"]:
            if c in df.columns:
                df = df.rename(columns={c: "prob_smish"})
                break
    # Ensure label is integer
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df

# Draw a simple confusion matrix image using counts
def confusion_matrix_figure(tn, fp, fn, tp, title: str, out_path: Path):
    mat = np.array([[tn, fp],
                    [fn, tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(mat, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Ham", "Smish"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Ham", "Smish"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# Draw ROC curve and save AUC in the legend
def roc_curve_figure(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# Draw PR curve and print PR-AUC in the legend
def pr_curve_figure(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(prec[::-1], rec[::-1])  # area estimate via trapezoid rule
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(rec, prec, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# Make a tag safe for filenames
def sanitize_tag(tag: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", tag)

# Find all report JSON files to compare
report_files = sorted(REPORTS.glob("*_report.json"))
if not report_files:
    raise SystemExit("No reports/*_report.json found. Train models first.")

# Accumulators for summary tables
rows_metrics = []
rows_leader = []

# Walk through each model report and collect metrics and figures
for rpt in report_files:
    J = load_json(rpt)

    # Decide a model tag for display and filenames
    tag = J.get("model") or J.get("run") or rpt.stem.replace("_report", "")
    tag = sanitize_tag(tag)

    # Pull threshold and metrics blocks
    test = J.get("metrics", {}).get("test", {})
    testd = J.get("metrics", {}).get("test_deploy", {})
    thr = testd.get("threshold", test.get("threshold"))

    # Skip models missing required values
    needed = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]
    if not all(k in testd for k in needed) or thr is None:
        print(f"[warn] Skipping {tag}: missing metrics or threshold in JSON.")
        continue

    # Save confusion matrix image from JSON counts
    cm_png = FIG_DIR / f"cm_{tag}_test_deploy.png"
    confusion_matrix_figure(
        int(testd["tn"]), int(testd["fp"]),
        int(testd["fn"]), int(testd["tp"]),
        title=f"Confusion Matrix — {tag} (test_deploy)",
        out_path=cm_png
    )

    # Draw ROC and PR curves if a predictions CSV exists
    pred = try_read_predictions(tag, "test_deploy")
    if pred is not None and {"label", "prob_smish"}.issubset(pred.columns):
        y = pred["label"].to_numpy()
        p = pred["prob_smish"].to_numpy(dtype=float)
        roc_curve_figure(y, p, title=f"ROC — {tag} (test_deploy)",
                         out_path=FIG_DIR / f"roc_{tag}_test_deploy.png")
        pr_curve_figure(y, p, title=f"PR — {tag} (test_deploy)",
                        out_path=FIG_DIR / f"pr_{tag}_test_deploy.png")
    else:
        print(f"[info] No predictions CSV for {tag}; ROC/PR curves skipped.")

    # Add test split metrics row
    rows_metrics.append({
        "model": tag, "split": "test",
        "threshold": test.get("threshold", np.nan),
        "accuracy": test.get("accuracy", np.nan),
        "precision": test.get("precision", np.nan),
        "recall": test.get("recall", np.nan),
        "f1": test.get("f1", np.nan),
        "roc_auc": test.get("roc_auc", np.nan),
        "pr_auc": test.get("pr_auc", np.nan),
        "tn": test.get("tn", np.nan), "fp": test.get("fp", np.nan),
        "fn": test.get("fn", np.nan), "tp": test.get("tp", np.nan),
    })

    # Add test_deploy split metrics row
    rows_metrics.append({
        "model": tag, "split": "test_deploy",
        "threshold": thr,
        "accuracy": testd["accuracy"],
        "precision": testd["precision"],
        "recall": testd["recall"],
        "f1": testd["f1"],
        "roc_auc": testd["roc_auc"],
        "pr_auc": testd["pr_auc"],
        "tn": testd["tn"], "fp": testd["fp"],
        "fn": testd["fn"], "tp": testd["tp"],
    })

# Build metrics DataFrame
if not rows_metrics:
    raise SystemExit("No complete metrics found to compare.")
dfm = pd.DataFrame(rows_metrics)

# Build leaderboard on test_deploy sorted by PR-AUC then F1 then Recall
lead = (
    dfm[dfm["split"] == "test_deploy"]
    .sort_values(["pr_auc", "f1", "recall"], ascending=False)
    .reset_index(drop=True)
)

# Save tables as CSV
dfm.to_csv(CSV_DIR / "model_metrics_summary.csv", index=False)
lead.to_csv(CSV_DIR / "model_leaderboard.csv", index=False)

# Print a short top five view for quick inspection
print("\n=== Leaderboard (test_deploy) — Top 5 by PR-AUC → F1 → Recall ===")
print(lead[["model","threshold","accuracy","precision","recall","f1","roc_auc","pr_auc"]].head(5).to_string(index=False))

# Make a bar chart comparing F1 and PR-AUC for test_deploy
fig, ax = plt.subplots(figsize=(9, 4.8))
X = np.arange(len(lead))
width = 0.38
ax.bar(X - width/2, lead["f1"].values, width, label="F1")
ax.bar(X + width/2, lead["pr_auc"].values, width, label="PR-AUC")
ax.set_xticks(X)
ax.set_xticklabels(lead["model"].tolist(), rotation=30, ha="right")
ax.set_ylabel("Score")
ax.set_title("Model Comparison on test_deploy")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "bars_test_deploy_f1_prauc.png", dpi=160)
plt.close(fig)

# Print final saved file locations
print(f"\nSaved:")
print(f" - Leaderboard → {CSV_DIR/'model_leaderboard.csv'}")
print(f" - Metrics     → {CSV_DIR/'model_metrics_summary.csv'}")
print(f" - Figures     → {FIG_DIR}/ (confusion matrices, ROC, PR curves, bars)")
