#!/usr/bin/env python3
# scripts/run_bert.py

# Train and evaluate BERT base uncased for smishing detection following the project protocol.
# Train on the train_balanced split.
# Choose one decision threshold on val_deploy that gives the best F1 score.
# Evaluate on test and test_deploy using that fixed threshold.
# Use class weighted cross entropy and select the compute device automatically.
#
# INPUTS
# data/processed/train_balanced.parquet
# data/processed/val_deploy.parquet
# data/processed/test.parquet
# data/processed/test_deploy.parquet
#
# OUTPUTS
# models/bert_base                       saved model and tokenizer
# reports/bert_report.json               summary with settings and metrics
# reports/csv/baseline_train_log.csv     appended class counts for training
# reports/csv/baseline_metrics_log.csv   appended metrics for test splits
# reports/predictions_bert_{val_deploy,test,test_deploy}_with_probs.csv  probabilities

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

# PARAMETERS
# Fixed values so runs are repeatable and easy to compare
MODEL_NAME    = "bert-base-uncased"
SEED          = 1337
EPOCHS        = 3
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
MAX_LENGTH    = 160
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.10
GRAD_CLIP_NORM= 1.0
LOG_EVERY_STEPS = 50

# PATHS
# Keep outputs under models and reports for simple discovery and cleanup
PROC    = Path("data/processed")
MODELS  = Path("models");  MODELS.mkdir(parents=True, exist_ok=True)
OUT_DIR = MODELS / "bert_base"
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPORTS / "csv"; CSV_DIR.mkdir(parents=True, exist_ok=True)

# IO HELPERS
# Append one row to a CSV file. Create the header on first write.
def write_csv_row(path: Path, row: dict, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if header is None:
        header = list(row.keys())
    with path.open("a", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

# Write a small JSON file with indentation
def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2))

# Save probabilities for later analysis
def save_preds(tag: str, split: str, y: np.ndarray, p: np.ndarray):
    df = pd.DataFrame({"label": y.astype(int), "prob_smish": p.astype(float)})
    df.to_csv(REPORTS / f"predictions_{tag}_{split}_with_probs.csv", index=False)

# DEVICE
# Prefer CUDA, then Apple MPS, then CPU
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

# DATA
# Load a split and enforce required columns and types
def load_split(name: str) -> pd.DataFrame:
    p = PROC / f"{name}.parquet"
    if not p.exists():
        raise SystemExit(f"Missing {p}.")
    df = pd.read_parquet(p)
    need = {"text", "label"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{p} must contain columns: text, label")
    return df.dropna(subset=["text", "label"]).assign(
        text=lambda d: d["text"].astype(str),
        label=lambda d: d["label"].astype(int)
    )

# Minimal PyTorch dataset that tokenizes on access
class ParquetTextDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=160):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.maxlen = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        # Padding is handled by the data collator for efficient batches
        enc = self.tok(self.texts[idx], truncation=True, max_length=self.maxlen, padding=False)
        enc["labels"] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in enc.items()}

# EVALUATION
# Convert logits to probability of the positive class
def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    logits_t = torch.tensor(logits)
    probs = torch.softmax(logits_t, dim=-1)[:, 1].cpu().numpy()
    return probs

# Choose the probability threshold that maximizes F1 on validation
def f1_optimal_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    thr = np.r_[0.0, thr]  # align with precision and recall arrays
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(f1s))])

# Compute metrics at a given threshold
def eval_at_threshold(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # ROC and average precision use probabilities and do not depend on the threshold
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(ap),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# MAIN
def main():
    # Set seeds for consistent initialization and data order
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Select device and decide on mixed precision
    device = pick_device()
    use_cuda = device == "cuda"
    print(f"[info] device: {device}")

    # Load the four protocol splits
    train_df  = load_split("train_balanced")
    valdep_df = load_split("val_deploy")
    test_df   = load_split("test")
    tdep_df   = load_split("test_deploy")

    # Build tokenizer and datasets
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds_train = ParquetTextDataset(train_df, tok, MAX_LENGTH)
    ds_vdep  = ParquetTextDataset(valdep_df, tok, MAX_LENGTH)
    ds_test  = ParquetTextDataset(test_df, tok, MAX_LENGTH)
    ds_tdep  = ParquetTextDataset(tdep_df, tok, MAX_LENGTH)

    # Compute class weights to address label imbalance
    y_train = train_df["label"].values
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = torch.tensor(weights, dtype=torch.float32)

    # Create model with two classes
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # Trainer that applies class weights during the loss computation
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: int | None = None):
            labels = inputs.pop("labels")
            if labels.dtype != torch.long:
                labels = labels.long()
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Training arguments
    # Disable periodic evaluation and periodic saving and save once at the end
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=GRAD_CLIP_NORM,
        logging_steps=LOG_EVERY_STEPS,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        no_cuda=not use_cuda,
        fp16=use_cuda,
        seed=SEED,
        report_to="none",
    )

    # Pad to the longest item in each batch for speed and memory efficiency
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    # Assemble trainer
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        tokenizer=tok,
        data_collator=data_collator,
    )

    # Fit the classifier on the balanced training split
    trainer.train()

    # Save model and tokenizer for reuse
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    # Select a threshold on the validation split
    pred_vdep = trainer.predict(ds_vdep)
    p_vd = probs_from_logits(pred_vdep.predictions)
    y_vd = pred_vdep.label_ids
    thr  = f1_optimal_threshold(y_vd, p_vd)

    # Evaluate on test and test_deploy using the fixed threshold
    pred_test  = trainer.predict(ds_test)
    pred_tdep  = trainer.predict(ds_tdep)

    p_te = probs_from_logits(pred_test.predictions);  y_te = pred_test.label_ids
    p_td = probs_from_logits(pred_tdep.predictions);  y_td = pred_tdep.label_ids

    res_test = eval_at_threshold(y_te, p_te, thr)
    res_td   = eval_at_threshold(y_td, p_td, thr)

    # Save probabilities for later analysis
    save_preds("bert", "val_deploy", y_vd, p_vd)
    save_preds("bert", "test",       y_te, p_te)
    save_preds("bert", "test_deploy",y_td, p_td)

    # Log training distribution
    train_row = {
        "run": "bert",
        "train_split": "train_balanced",
        "total": int(len(train_df)),
        "ham": int((train_df.label == 0).sum()),
        "smish": int((train_df.label == 1).sum()),
        "smish_pct": round(train_df.label.mean() * 100.0, 3),
        "ngram_max": "",
        "min_df": "",
        "max_df": "",
        "C": "",
        "alpha": "",
        "seed": SEED,
        "class_weight": "balanced",
    }
    write_csv_row(
        CSV_DIR / "baseline_train_log.csv",
        train_row,
        header=[
            "run","train_split","total","ham","smish","smish_pct",
            "ngram_max","min_df","max_df","C","alpha","seed","class_weight"
        ],
    )

    # Log metrics for both evaluation splits
    for split_name, res in [("test", res_test), ("test_deploy", res_td)]:
        metrics_row = {
            "split": split_name, "model": "bert", "threshold": res["threshold"],
            "accuracy": res["accuracy"], "precision": res["precision"], "recall": res["recall"],
            "f1": res["f1"], "roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"],
            "tn": res["tn"], "fp": res["fp"], "fn": res["fn"], "tp": res["tp"],
            "ngram_max": "", "min_df": "", "max_df": "", "C": "", "alpha": "", "seed": SEED,
            "train_split": "train_balanced",
        }
        write_csv_row(
            CSV_DIR / "baseline_metrics_log.csv",
            metrics_row,
            header=[
                "split","model","threshold","accuracy","precision","recall","f1",
                "roc_auc","pr_auc","tn","fp","fn","tp",
                "ngram_max","min_df","max_df","C","alpha","seed","train_split"
            ],
        )

    # Write one JSON file with settings, device, threshold, and metrics
    save_json(
        {
            "model": "bert",
            "pretrained_name": MODEL_NAME,
            "train": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "max_length": MAX_LENGTH,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "seed": SEED,
                "class_weight": weights.tolist(),
                "counts": {
                    "total": int(len(train_df)),
                    "ham": int((train_df.label == 0).sum()),
                    "smish": int((train_df.label == 1).sum()),
                    "smish_pct": float(train_df.label.mean() * 100.0),
                },
            },
            "device": device,
            "threshold": thr,
            "metrics": {"test": res_test, "test_deploy": res_td},
        },
        REPORTS / "bert_report.json",
    )

    # Short status so expected files are easy to confirm
    print(f"[done] models → {OUT_DIR}")
    print("reports → reports/bert_report.json")
    print("logs → reports/csv/{baseline_train_log.csv,baseline_metrics_log.csv}")
    print("preds → reports/predictions_bert_{val_deploy,test,test_deploy}_with_probs.csv")

if __name__ == "__main__":
    main()
