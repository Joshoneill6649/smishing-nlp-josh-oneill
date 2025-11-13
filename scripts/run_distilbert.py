#!/usr/bin/env python3
# scripts/run_distilbert.py (preproc 160)
#
# Train and evaluate DistilBERT base uncased for smishing detection following the project protocol.
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
# models/distilbert_base                          saved model and tokenizer
# reports/distilbert_report.json                  summary with settings and metrics
# reports/csv/baseline_train_log.csv              appended class counts for training
# reports/csv/baseline_metrics_log.csv            appended metrics for test splits
# reports/predictions_distilbert_{val_deploy,test,test_deploy}_with_probs.csv  probabilities

from pathlib import Path
import json, csv
import numpy as np
import pandas as pd
import torch

# Metrics and utilities from scikit-learn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

# Hugging Face Transformers: tokenizer, model, Trainer stack, collator, callback
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    TrainerCallback
)

# Fixed hyperparameters for repeatable runs
MODEL_NAME      = "distilbert-base-uncased"
SEED            = 1337
EPOCHS          = 3
BATCH_SIZE      = 16
LEARNING_RATE   = 2e-5
MAX_LENGTH      = 160
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.10
GRAD_CLIP_NORM  = 1.0
LOG_EVERY_STEPS = 50  # how often to print loss during training

# Project paths and output folders
PROC    = Path("data/processed")
MODELS  = Path("models");  MODELS.mkdir(parents=True, exist_ok=True)
OUT_DIR = MODELS / "distilbert_base"
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPORTS / "csv"; CSV_DIR.mkdir(parents=True, exist_ok=True)

# Append one row to a CSV file (create header on first write)
def write_csv_row(path: Path, row: dict, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if header is None: header = list(row.keys())
    with path.open("a", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row)

# Save a JSON dictionary with indentation
def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2))

# Save labels and probabilities for later analysis
def save_preds(tag: str, split: str, y: np.ndarray, p: np.ndarray):
    df = pd.DataFrame({"label": y.astype(int), "prob_smish": p.astype(float)})
    df.to_csv(REPORTS / f"predictions_{tag}_{split}_with_probs.csv", index=False)

# Pick the best available device: CUDA, then MPS, then CPU
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

# Load a parquet split and enforce required columns and types
def load_split(name: str) -> pd.DataFrame:
    p = PROC / f"{name}.parquet"
    if not p.exists(): raise SystemExit(f"Missing {p}.")
    df = pd.read_parquet(p)
    need = {"text","label"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{p} must contain columns: text, label")
    return df.dropna(subset=["text","label"]).assign(
        text=lambda d: d["text"].astype(str),
        label=lambda d: d["label"].astype(int)
    )

# Simple PyTorch dataset that tokenizes text when items are accessed
class ParquetTextDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=160):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.maxlen = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        # Padding is deferred to the data collator for efficient batches
        enc = self.tok(self.texts[idx], truncation=True, max_length=self.maxlen, padding=False)
        enc["labels"] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in enc.items()}

# Convert raw logits to positive class probabilities
def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    logits_t = torch.tensor(logits)
    probs = torch.softmax(logits_t, dim=-1)[:, 1].cpu().numpy()
    return probs

# Pick a probability threshold that maximizes F1 on validation
def f1_optimal_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    thr = np.r_[0.0, thr]  # align with prec/rec arrays
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(f1s))])

# Compute common metrics at a fixed threshold
def eval_at_threshold(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else float("nan")
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true))==2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "threshold": float(thr), "accuracy": float(acc), "precision": float(prec),
        "recall": float(rec), "f1": float(f1), "roc_auc": float(roc), "pr_auc": float(ap),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

# Lightweight console progress during training
class PrintCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"[train] starting — epochs={args.num_train_epochs}, batch_size={args.per_device_train_batch_size}, lr={args.learning_rate}")
    def on_epoch_begin(self, args, state, control, **kwargs):
        ep = int(state.epoch) + 1 if getattr(state, "epoch", None) is not None else "?"
        print(f"[train] epoch {ep} begin…")
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = getattr(state, "global_step", None)
            loss = logs["loss"]
            print(f"[step {step}] loss={loss:.4f}")
    def on_epoch_end(self, args, state, control, **kwargs):
        ep = int(state.epoch) if getattr(state, "epoch", None) is not None else "?"
        print(f"[train] epoch {ep} done.")

# Main training and evaluation entry point
def main():
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Choose device and whether fp16 can be enabled
    device = pick_device()
    use_cuda = (device == "cuda")
    print(f"[info] device: {device}")

    # Load protocol splits
    train_df  = load_split("train_balanced")
    valdep_df = load_split("val_deploy")
    test_df   = load_split("test")
    tdep_df   = load_split("test_deploy")

    # Build tokenizer and datasets
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)  # pulls tokenizer from the HF Hub or local cache
    ds_train = ParquetTextDataset(train_df, tok, MAX_LENGTH)
    ds_vdep  = ParquetTextDataset(valdep_df, tok, MAX_LENGTH)
    ds_test  = ParquetTextDataset(test_df, tok, MAX_LENGTH)
    ds_tdep  = ParquetTextDataset(tdep_df, tok, MAX_LENGTH)

    # Compute class weights for imbalanced labels
    y_train = train_df["label"].values
    classes = np.array([0,1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = torch.tensor(weights, dtype=torch.float32)

    # Create sequence classification model with two labels
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # pulls model from the HF Hub or local cache
    model.to(device)

    # Trainer subclass that applies class-weighted CrossEntropy
    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
        ):
            labels = inputs.pop("labels")
            if labels.dtype != torch.long:
                labels = labels.long()
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Trainer configuration (no mid-epoch eval or checkpointing)
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
        fp16=use_cuda,   # fp16 only on CUDA; ignored on CPU/MPS
        seed=SEED,
        report_to="none",
    )

    # Pad each batch to the longest item for efficiency
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    # Assemble Trainer with datasets and callback
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        tokenizer=tok,
        data_collator=data_collator,
        callbacks=[PrintCallback()],
    )

    # Fine tune the model
    trainer.train()

    # Save model and tokenizer to the output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    # Predict on validation and choose F1-max threshold
    pred_vdep = trainer.predict(ds_vdep)
    p_vd = probs_from_logits(pred_vdep.predictions)
    y_vd = pred_vdep.label_ids
    thr  = f1_optimal_threshold(y_vd, p_vd)

    # Predict on test and test_deploy with that fixed threshold
    pred_test  = trainer.predict(ds_test)
    pred_tdep  = trainer.predict(ds_tdep)

    p_te = probs_from_logits(pred_test.predictions);  y_te = pred_test.label_ids
    p_td = probs_from_logits(pred_tdep.predictions);  y_td = pred_tdep.label_ids

    # Compute metrics for both splits
    res_test = eval_at_threshold(y_te, p_te, thr)
    res_td   = eval_at_threshold(y_td, p_td, thr)

    # Save probabilities for later analysis
    save_preds("distilbert", "val_deploy", y_vd, p_vd)
    save_preds("distilbert", "test",       y_te, p_te)
    save_preds("distilbert", "test_deploy",y_td, p_td)

    # Append one training distribution row
    train_row = {
        "run":"distilbert", "train_split":"train_balanced",
        "total": int(len(train_df)),
        "ham": int((train_df.label==0).sum()),
        "smish": int((train_df.label==1).sum()),
        "smish_pct": round(train_df.label.mean()*100.0, 3),
        "ngram_max":"", "min_df":"", "max_df":"",
        "C":"", "alpha":"", "seed": SEED, "class_weight":"balanced"
    }
    write_csv_row(CSV_DIR/"baseline_train_log.csv", train_row,
                  header=["run","train_split","total","ham","smish","smish_pct",
                          "ngram_max","min_df","max_df","C","alpha","seed","class_weight"])

    # Append metrics rows for test and test_deploy
    for split_name, res in [("test", res_test), ("test_deploy", res_td)]:
        metrics_row = {
            "split": split_name, "model": "distilbert", "threshold": res["threshold"],
            "accuracy": res["accuracy"], "precision": res["precision"], "recall": res["recall"],
            "f1": res["f1"], "roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"],
            "tn": res["tn"], "fp": res["fp"], "fn": res["fn"], "tp": res["tp"],
            "ngram_max":"", "min_df":"", "max_df":"", "C":"", "alpha":"", "seed": SEED, "train_split":"train_balanced"
        }
        write_csv_row(CSV_DIR/"baseline_metrics_log.csv", metrics_row,
                      header=["split","model","threshold","accuracy","precision","recall","f1",
                              "roc_auc","pr_auc","tn","fp","fn","tp",
                              "ngram_max","min_df","max_df","C","alpha","seed","train_split"])

    # Write a single JSON report with settings, device, threshold, and metrics
    save_json({
        "model": "distilbert",
        "pretrained_name": MODEL_NAME,
        "train": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
            "max_length": MAX_LENGTH, "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO, "grad_clip_norm": GRAD_CLIP_NORM,
            "seed": SEED, "class_weight": weights.tolist(),
            "counts": {"total": int(len(train_df)),
                       "ham": int((train_df.label==0).sum()),
                       "smish": int((train_df.label==1).sum()),
                       "smish_pct": float(train_df.label.mean()*100.0)}
        },
        "device": device,
        "threshold": thr,
        "metrics": {"test": res_test, "test_deploy": res_td}
    }, REPORTS/"distilbert_report.json")

    # Short status so outputs are easy to verify
    print(f"\n[done] DistilBERT trained & evaluated. (thr={thr:.4f})")
    print("Model  → models/distilbert_base/")
    print("Logs   → reports/csv/baseline_train_log.csv, reports/csv/baseline_metrics_log.csv")
    print("JSON   → reports/distilbert_report.json")
    print("Preds  → reports/predictions_distilbert_{val_deploy,test,test_deploy}_with_probs.csv")

if __name__ == "__main__":
    main()
