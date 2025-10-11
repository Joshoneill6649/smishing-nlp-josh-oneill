#!/usr/bin/env python3
# scripts/03_split.py
"""
Create stratified splits from data/processed/combined.parquet.

Outputs (to data/processed/):
- train.parquet, val.parquet, test.parquet                (stratified 80/10/10)
- train_balanced.parquet                                  (50/50 undersample)
- val_deploy.parquet, test_deploy.parquet                 (~90:10 ham:smish by downsampling)

CLI options:
  --train 0.8 --val 0.1 --test 0.1 --seed 1337 --deploy_ratio 0.10
"""

from pathlib import Path
import argparse
import math
import pandas as pd
from sklearn.model_selection import train_test_split

PROC = Path("data/processed")
INP  = PROC / "combined.parquet"

def summarize(name, df):
    n = len(df)
    ham = int((df.label == 0).sum())
    sm  = int((df.label == 1).sum())
    pct = (df.label.mean() * 100) if n else float("nan")
    return f"{name:15s} N={n:7,d} | ham={ham:7,d} | smish={sm:7,d} | smish%={pct:6.2f}%"

def make_deploy_split(df_split: pd.DataFrame, target_smish_ratio=0.10, seed=1337):
    """
    Downsample within a split to approx target ham:smish ratio (default 90:10).
    Never upsamples; clips to availability.
    """
    if df_split.empty:
        return df_split.copy()

    sm = df_split[df_split.label == 1]
    hm = df_split[df_split.label == 0]

    # Ideal counts at current split size:
    n = len(df_split)
    ideal_sm = int(round(target_smish_ratio * n))
    ideal_hm = n - ideal_sm

    # Clip to what's available; if too few of one class, take all available of the limiting class
    n_sm = min(ideal_sm, len(sm))
    n_hm = min(ideal_hm, len(hm))

    # If either is zero, fallback to taking what we have with same global ratio
    if n_sm == 0 or n_hm == 0:
        # Try to carve out as much as possible with target ratio using the limiting class
        if len(sm) == 0 or len(hm) == 0:
            # Can't make a deploy ratio here; just return original split
            return df_split.copy()
        # Recompute with limiting class as anchor
        # If smish is limiting:
        if len(sm) < len(hm):
            n_sm = len(sm)
            n_hm = min(int((1 - target_smish_ratio) * n_sm / target_smish_ratio), len(hm))
        else:
            n_hm = len(hm)
            n_sm = min(int(target_smish_ratio * n_hm / (1 - target_smish_ratio)), len(sm))

    out = pd.concat([
        sm.sample(n=n_sm, random_state=seed, replace=False),
        hm.sample(n=n_hm, random_state=seed, replace=False),
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val",   type=float, default=0.10)
    ap.add_argument("--test",  type=float, default=0.10)
    ap.add_argument("--seed",  type=int,   default=1337)
    ap.add_argument("--deploy_ratio", type=float, default=0.10, help="target smish ratio for deploy splits (default 0.10)")
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "train+val+test must sum to 1.0"

    if not INP.exists():
        raise SystemExit(f"Missing {INP}. Run normalize_merge_english.py first.")

    df = pd.read_parquet(INP)
    if not {"text", "label", "source"}.issubset(df.columns):
        raise SystemExit("combined.parquet must have columns: text, label, source")

    # (safety) ensure no NaNs and non-empty texts
    df = df.dropna(subset=["text", "label"]).copy()
    df = df[df["text"].astype(str).str.len() > 0]

    # Stratified split 80/10/10
    test_size = args.val + args.test
    train, temp = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=args.seed
    )
    # Now split temp into val/test equally per requested proportions
    rel_test = args.test / (args.val + args.test) if (args.val + args.test) > 0 else 0.5
    val, test = train_test_split(
        temp, test_size=rel_test, stratify=temp["label"], random_state=args.seed
    )

    # Balanced train (50/50) by undersampling majority
    n_ham = (train.label == 0).sum()
    n_sm  = (train.label == 1).sum()
    m = int(min(n_ham, n_sm))
    train_bal = pd.concat([
        train[train.label == 0].sample(n=m, random_state=args.seed, replace=False),
        train[train.label == 1].sample(n=m, random_state=args.seed, replace=False),
    ]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Deploy 90:10 (or custom) by downsampling inside each split
    val_deploy  = make_deploy_split(val,  target_smish_ratio=args.deploy_ratio, seed=args.seed)
    test_deploy = make_deploy_split(test, target_smish_ratio=args.deploy_ratio, seed=args.seed)

    # Write outputs
    PROC.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROC/"train.parquet", index=False)
    val.to_parquet(PROC/"val.parquet", index=False)
    test.to_parquet(PROC/"test.parquet", index=False)
    train_bal.to_parquet(PROC/"train_balanced.parquet", index=False)
    val_deploy.to_parquet(PROC/"val_deploy.parquet", index=False)
    test_deploy.to_parquet(PROC/"test_deploy.parquet", index=False)

    print("\n=== Split Summary ===")
    for name, d in [
        ("train", train), ("val", val), ("test", test),
        ("train_balanced", train_bal), ("val_deploy", val_deploy), ("test_deploy", test_deploy)
    ]:
        print(summarize(name, d))

    # By-source heads for quick sanity
    print("\n=== By-source (train) ===")
    print(train["source"].value_counts().head())
    print("\n=== By-source (val) ===")
    print(val["source"].value_counts().head())
    print("\n=== By-source (test) ===")
    print(test["source"].value_counts().head())

if __name__ == "__main__":
    main()
