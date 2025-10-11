#!/usr/bin/env python3
"""
Train AND evaluate DistilBERT for smishing detection (live training updates).

- Train on:        data/processed/train.parquet
- Evaluate on:     data/processed/test.parquet, data/processed/test_deploy.parquet
- Threshold:       0.5
- Imbalance:       class-weighted CrossEntropy
- Device:          auto (cuda -> mps -> cpu)
- Logs during training: step loss, epoch begin/end

Artifacts (overwrite each run):
  models/distilbert_base/        (HF model+tokenizer)
  reports/distilbert_report.json
Appends rows to shared logs:
  reports/csv/baseline_train_log.csv
  reports/csv/baseline_metrics_log.csv
"""

from pathlib import Path
import json, csv
import numpy as np
import pandas as pd
import torch

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    TrainerCallback
)

# ---------- fixed hyperparams ----------
MODEL_NAME    = "distilbert-base-uncased"
SEED          = 1337
EPOCHS        = 3
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
MAX_LENGTH    = 160
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.10
GRAD_CLIP_NORM= 1.0
THRESHOLD     = 0.5
LOG_EVERY_STEPS = 50  # how often to print loss during training

# ---------- paths ----------
PROC    = Path("data/processed")
MODELS  = Path("models");  MODELS.mkdir(parents=True, exist_ok=True)
OUT_DIR = MODELS / "distilbert_base"
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPORTS / "csv"; CSV_DIR.mkdir(parents=True, exist_ok=True)

# ---------- tiny io helpers ----------
def write_csv_row(path: Path, row: dict, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if header is None: header = list(row.keys())
    with path.open("a", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row)

def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2))

# ---------- device pick (auto) ----------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

# ---------- data utils ----------
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

class ParquetTextDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=160):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.maxlen = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, max_length=self.maxlen, padding=False)
        enc["labels"] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in enc.items()}

# ---------- eval helpers ----------
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

def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    logits_t = torch.tensor(logits)
    probs = torch.softmax(logits_t, dim=-1)[:, 1].cpu().numpy()
    return probs

# ---------- live logging callback ----------
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

# ---------- main ----------
def main():
    # seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # device
    device = pick_device()
    use_cuda = (device == "cuda")
    print(f"[info] device: {device}")

    # load splits
    train_df = load_split("train")
    test_df  = load_split("test")
    tdep_df  = load_split("test_deploy")

    # tokenizer & datasets
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds_train = ParquetTextDataset(train_df, tok, MAX_LENGTH)
    ds_test  = ParquetTextDataset(test_df, tok, MAX_LENGTH)
    ds_tdep  = ParquetTextDataset(tdep_df, tok, MAX_LENGTH)

    # class weights (balanced)
    y_train = train_df["label"].values
    classes = np.array([0,1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = torch.tensor(weights, dtype=torch.float32)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # Trainer override compatible with transformers 4.57 (passes num_items_in_batch)
    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,  # accept new arg
        ):
            labels = inputs.pop("labels")
            if labels.dtype != torch.long:
                labels = labels.long()
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Training arguments (transformers 4.57 uses eval_strategy)
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
        fp16=use_cuda,   # fp16 only on CUDA; on CPU it is ignored
        seed=SEED,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        tokenizer=tok,
        data_collator=data_collator,
        callbacks=[PrintCallback()],
    )

    # train (prints step loss + epoch markers)
    trainer.train()

    # save model/tokenizer
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    # evaluate (post-train) on test and test_deploy
    pred_test  = trainer.predict(ds_test)
    pred_tdep  = trainer.predict(ds_tdep)
    p_te = probs_from_logits(pred_test.predictions)
    p_td = probs_from_logits(pred_tdep.predictions)
    y_te = pred_test.label_ids
    y_td = pred_tdep.label_ids

    res_test = eval_at_threshold(y_te, p_te, THRESHOLD)
    res_td   = eval_at_threshold(y_td, p_td, THRESHOLD)

    # append to shared CSV logs (same format as baselines)
    train_row = {
        "run":"distilbert", "train_split":"train",
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

    for split_name, res in [("test", res_test), ("test_deploy", res_td)]:
        metrics_row = {
            "split": split_name, "model": "distilbert", "threshold": res["threshold"],
            "accuracy": res["accuracy"], "precision": res["precision"], "recall": res["recall"],
            "f1": res["f1"], "roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"],
            "tn": res["tn"], "fp": res["fp"], "fn": res["fn"], "tp": res["tp"],
            "ngram_max":"", "min_df":"", "max_df":"", "C":"", "alpha":"", "seed": SEED, "train_split":"train"
        }
        write_csv_row(CSV_DIR/"baseline_metrics_log.csv", metrics_row,
                      header=["split","model","threshold","accuracy","precision","recall","f1",
                              "roc_auc","pr_auc","tn","fp","fn","tp",
                              "ngram_max","min_df","max_df","C","alpha","seed","train_split"])

    # JSON report
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
        "threshold": THRESHOLD,
        "metrics": {"test": res_test, "test_deploy": res_td}
    }, REPORTS/"distilbert_report.json")

    print("\n[done] DistilBERT trained & evaluated.")
    print("Model  → models/distilbert_base/")
    print("Logs   → reports/csv/baseline_train_log.csv, reports/csv/baseline_metrics_log.csv")
    print("JSON   → reports/distilbert_report.json")

if __name__ == "__main__":
    main()
