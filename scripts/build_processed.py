#!/usr/bin/env python3
# Josh O’Neill • x23315369 — build processed dataset (clean, dedupe, stratify)

import os, re, json, hashlib, unicodedata, pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ROOT = pathlib.Path(".")
RAW = ROOT/"data/raw"
PRO = ROOT/"data/processed"
FIG = ROOT/"reports/figures"
(PRO/"splits").mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# load already-normalised raw CSVs (text,label)
sources = []
def load(name, source_id):
    p = RAW/name
    if not p.exists():
        print(f"(skip) {name} not found at {p}")
        return
    df = pd.read_csv(p)
    assert set(df.columns) >= {"text","label"}, f"{name} missing columns"
    df["label"] = df["label"].astype(str).str.lower()
    df["source"] = source_id
    sources.append(df)
    print(f"loaded {name}: {len(df)} rows")

load("sms_spam_collection.csv", "sms_spam")  # mixed ham/smish
load("smishtank.csv", "smishtank")           # smish-only
load("spamdam.csv", "spamdam")               # smish-only
assert sources, "No raw datasets found."

df = pd.concat(sources, ignore_index=True)

def clean_text(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = re.sub(r"(https?://\S+|www\.\S+)", "[URL]", s)
    s = re.sub(r"\s+", " ", s)
    return s

df["text_clean"] = df["text"].map(clean_text)
df["text_hash"]  = df["text_clean"].map(lambda s: hashlib.sha256(s.encode()).hexdigest())

before = len(df)
df = df.drop_duplicates(subset=["text_hash"]).reset_index(drop=True)
after = len(df)
print(f"dedup: {before} -> {after} (removed {before-after})")

assert df["label"].nunique()==2, "Combined must have both ham and smish."

SEED = 23315369
train_val, test = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label"]
)
val_ratio = 0.15 / 0.85
train, val = train_test_split(
    train_val, test_size=val_ratio, random_state=SEED, stratify=train_val["label"]
)

for name, part in [("combined", df), ("train", train), ("val", val), ("test", test)]:
    part.to_parquet(PRO/f"{name}.parquet", index=False)
    print(f"saved {PRO/f'{name}.parquet'} rows={len(part)}")

splits = {
    "seed": SEED,
    "sizes": {k: len(v) for k, v in {"combined":df,"train":train,"val":val,"test":test}.items()},
    "counts": {k: v["label"].value_counts().to_dict() for k, v in {"combined":df,"train":train,"val":val,"test":test}.items()},
}
with open(PRO/"splits"/"combined_splits.json","w") as f:
    json.dump(splits, f, indent=2)
print(f"saved {PRO/'splits'/'combined_splits.json'}")

# class-balance plot (combined)
counts = df["label"].value_counts().sort_index()
plt.figure(figsize=(4,3))
counts.plot(kind="bar")
plt.title("Class balance (combined)")
plt.xlabel("label"); plt.ylabel("count")
plt.tight_layout()
plot_path = FIG/"step1_class_balance.png"
plt.savefig(plot_path, dpi=160)
print(f"saved figure {plot_path}")

def summary(name, part):
    vc = part["label"].value_counts().to_dict()
    print(f"{name:8s} rows={len(part):6d}  mix={vc}")
for name, part in [("combined", df), ("train", train), ("val", val), ("test", test)]:
    summary(name, part)