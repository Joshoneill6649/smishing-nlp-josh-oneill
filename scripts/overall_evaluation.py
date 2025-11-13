#!/usr/bin/env python3
# Model performance overall for thesis Evaluation and Conclusion.
# Reads: reports/csv/baseline_metrics_log.csv
# Produces in reports/overall_evaluation/:
#   - model_leaderboard.csv
#   - model_leaderboard_table.png      (leaderboard)
#   - f1_test_deploy_zoomed.png        (F1 bars)
#   - gap_to_best_f1_test_deploy.png   (F1(best) - F1(model) bars)

from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Set root directory and input/output paths
ROOT = Path(".")
CSV_PATH = ROOT / "reports" / "csv" / "baseline_metrics_log.csv"
OUT_DIR = ROOT / "reports" / "overall_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# List of key models to include in the leaderboard
KEY_MODELS = [
    "tfidf_lr",
    "tfidf_linearsvc",
    "tfidf_mnb",
    "bert",
    "distilbert",
    "bert_preproc256_v1",
    "distilbert_preproc256_v1",
    "emotion_fusion_model_1",
    "emotion_fusion_tfidf_lsvc_1",
]


# Load the metrics CSV and validate that required columns exist
def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"Missing metrics CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    needed = {"split", "model", "f1"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"{csv_path} must contain at least columns: {needed}")
    return df


# Build a leaderboard DataFrame for the key models and save it as CSV
def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in KEY_MODELS:
        # Filter rows for this model
        df_m = df[df["model"] == m]
        if df_m.empty:
            continue

        # Take the best F1 entry for test and test_deploy splits
        row_test = df_m[df_m["split"] == "test"].sort_values("f1", ascending=False).head(1)
        row_tdep = df_m[df_m["split"] == "test_deploy"].sort_values("f1", ascending=False).head(1)
        if row_test.empty or row_tdep.empty:
            continue

        rt = row_test.iloc[0]
        rd = row_tdep.iloc[0]

        # Collect the key metrics for this model
        rows.append({
            "model": m,
            "f1_test": float(rt["f1"]),
            "f1_test_deploy": float(rd["f1"]),
            "precision_test_deploy": float(rd["precision"]),
            "recall_test_deploy": float(rd["recall"]),
            "roc_auc_test_deploy": float(rd.get("roc_auc", float("nan"))),
            "pr_auc_test_deploy": float(rd.get("pr_auc", float("nan"))),
            "threshold_test_deploy": float(rd.get("threshold", float("nan"))),
        })

    if not rows:
        raise SystemExit("No rows found for KEY_MODELS in baseline_metrics_log.csv")

    # Sort models by deployment F1 and write leaderboard CSV
    lb = pd.DataFrame(rows)
    lb = lb.sort_values("f1_test_deploy", ascending=False).reset_index(drop=True)
    lb.to_csv(OUT_DIR / "model_leaderboard.csv", index=False)
    return lb


# Plot a nicely formatted leaderboard table as a PNG
def plot_leaderboard_table(lb: pd.DataFrame):
    # Copy and rename columns for display
    disp = lb.copy()
    col_map = {
        "model": "Model",
        "f1_test": "F1 (test)",
        "f1_test_deploy": "F1 (test_deploy)",
        "precision_test_deploy": "Precision (deploy)",
        "recall_test_deploy": "Recall (deploy)",
        "roc_auc_test_deploy": "ROC-AUC (deploy)",
        "pr_auc_test_deploy": "PR-AUC (deploy)",
        "threshold_test_deploy": "Thr (deploy)",
    }
    disp = disp.rename(columns=col_map)

    # Round numeric values for cleaner display
    for c in disp.columns:
        if c != "Model":
            disp[c] = disp[c].map(lambda x: f"{x:.4f}")

    n_rows, n_cols = disp.shape

    # Wider figure so it drops nicely into a report
    fig_width = 2 + 1.4 * n_cols
    fig_height = 1.5 + 0.4 * n_rows
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # Create the table object
    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        cellLoc="center",
        loc="center"
    )

    # Style header and body cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(9)
            cell.set_text_props(weight="bold")
        else:
            cell.set_fontsize(8)

    # Zebra stripes for rows (skip header row=0)
    for row in range(1, n_rows + 1):
        for col in range(n_cols):
            cell = table[row, col]
            if row % 2 == 1:
                cell.set_facecolor("#f5f5f5")  # light grey
            else:
                cell.set_facecolor("white")

    # Adjust column widths and add title
    table.auto_set_column_width(col=list(range(n_cols)))
    ax.set_title("Model leaderboard (test & deployment performance)", pad=8, fontsize=11)

    # Save the table as an image
    fig.tight_layout()
    out_path = OUT_DIR / "model_leaderboard_table.png"
    plt.savefig(out_path, dpi=250)
    plt.close()


# Plot a zoomed-in bar chart of F1 scores on the test_deploy split
def plot_f1_zoomed(lb: pd.DataFrame):
    models = lb["model"].tolist()
    f1s = lb["f1_test_deploy"].tolist()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, f1s)
    plt.title("F1 on deployment split (test_deploy)")
    plt.ylabel("F1")
    plt.xticks(rotation=35, ha="right")

    # Compute a tight y-axis range around the min and max F1 values
    min_f1 = min(f1s)
    max_f1 = max(f1s)
    margin = 0.005
    lower = max(0.95, min_f1 - margin)
    upper = min(1.0, max_f1 + margin)
    plt.ylim(lower, upper)

    # Annotate bars with their F1 values
    for b, v in zip(bars, f1s):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v + 0.0003,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Save the F1 bar chart
    plt.tight_layout()
    plt.savefig(OUT_DIR / "f1_test_deploy_zoomed.png", dpi=250)
    plt.close()


# Plot the gap from each model to the best deployment F1 score
def plot_gap_to_best(lb: pd.DataFrame):
    # Compute the best deployment F1 and each model's gap to that value
    best_f1 = lb["f1_test_deploy"].max()
    lb_gap = lb.copy()
    lb_gap["gap_to_best"] = best_f1 - lb_gap["f1_test_deploy"]

    models = lb_gap["model"].tolist()
    gaps = lb_gap["gap_to_best"].tolist()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, gaps)
    plt.axhline(0.0, linewidth=1)
    plt.title("Gap to best F1 on deployment split")
    plt.ylabel("F1(best) - F1(model)")
    plt.xticks(rotation=35, ha="right")

    # Annotate bars with their gap values
    for b, g in zip(bars, gaps):
        plt.text(
            b.get_x() + b.get_width() / 2,
            g + 0.0003,
            f"{g:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Save the gap to best bar chart
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gap_to_best_f1_test_deploy.png", dpi=250)
    plt.close()


# Main entry point that wires everything together
def main():
    # Load metrics and build leaderboard
    df = load_metrics(CSV_PATH)
    lb = build_leaderboard(df)

    # Print leaderboard to console
    print("\n[info] Model leaderboard (sorted by F1 on test_deploy):")
    print(lb.to_string(index=False))

    # Generate all plots
    plot_leaderboard_table(lb)
    plot_f1_zoomed(lb)
    plot_gap_to_best(lb)

    print(f"\n[done] Wrote artifacts in {OUT_DIR}")


# Run the script if executed directly
if __name__ == "__main__":
    main()
