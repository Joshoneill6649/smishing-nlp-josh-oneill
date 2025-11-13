#!/usr/bin/env python3
# scripts/split.py
# Create stratified train val test splits from data processed combined.parquet
# Also create a balanced train set by undersampling and deploy style splits with a target smish ratio
#
# INPUTS
# data/processed/combined.parquet with columns text label source
#
# OUTPUTS written to data processed
# train.parquet
# val.parquet
# test.parquet
# train_balanced.parquet       balanced 50 50 by undersampling
# val_deploy.parquet           about 90 10 ham to smish by downsampling only (could only get 15% on both)
# test_deploy.parquet          about 90 10 ham to smish by downsampling only

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
TEST_FRAC  = 0.10
SEED       = 1337
DEPLOY_RATIO = 0.10  # target smish ratio inside deploy splits, e.g., 0.10 means 90 10 ham to smish (ended up being 15)

# Paths to the processed folder and the combined parquet
PROC = Path("data/processed")
INP  = PROC / "combined.parquet"

# Format a one line summary of a split
def summarize(name, df):
    n = len(df)
    ham = int((df.label == 0).sum())
    sm  = int((df.label == 1).sum())
    pct = (df.label.mean() * 100) if n else float("nan")
    return f"{name:15s} N={n:7,d} | ham={ham:7,d} | smish={sm:7,d} | smish%={pct:6.2f}%"

# Downsample inside an existing split to reach an approximate target smish ratio
# No upsampling is used and counts are clipped to what is available
def make_deploy_split(df_split: pd.DataFrame, target_smish_ratio=DEPLOY_RATIO, seed=SEED):
    if df_split.empty:
        return df_split.copy()

    sm = df_split[df_split.label == 1]
    hm = df_split[df_split.label == 0]

    # Ideal counts at the current split size
    n = len(df_split)
    ideal_sm = int(round(target_smish_ratio * n))
    ideal_hm = n - ideal_sm

    # Clip to what is available
    n_sm = min(ideal_sm, len(sm))
    n_hm = min(ideal_hm, len(hm))

    # If one class is missing or too small fall back to a feasible ratio anchored on the limiting class
    if n_sm == 0 or n_hm == 0:
        if len(sm) == 0 or len(hm) == 0:
            return df_split.copy()
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
    # Validate split ratios sum to 1.0
    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-6, "train plus val plus test must sum to 1.0"

    # Ensure input exists and has the expected schema
    if not INP.exists():
        raise SystemExit(f"Missing {INP}. Run normalize_merge_english.py first.")

    df = pd.read_parquet(INP)
    if not {"text", "label", "source"}.issubset(df.columns):
        raise SystemExit("combined.parquet must have columns text label source")

    # Safety clean up in case of stray NaN or empty text
    df = df.dropna(subset=["text", "label"]).copy()
    df = df[df["text"].astype(str).str.len() > 0]

    # Stratified split into train and temp using the requested proportions
    test_size = VAL_FRAC + TEST_FRAC
    train, temp = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=SEED
    )

    # Split temp into val and test using the relative requested sizes
    rel_test = TEST_FRAC / (VAL_FRAC + TEST_FRAC) if (VAL_FRAC + TEST_FRAC) > 0 else 0.5
    val, test = train_test_split(
        temp, test_size=rel_test, stratify=temp["label"], random_state=SEED
    )

    # Balanced train 50 50 by undersampling the majority class
    n_ham = (train.label == 0).sum()
    n_sm  = (train.label == 1).sum()
    m = int(min(n_ham, n_sm))
    train_bal = pd.concat([
        train[train.label == 0].sample(n=m, random_state=SEED, replace=False),
        train[train.label == 1].sample(n=m, random_state=SEED, replace=False),
    ]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Make deploy style splits by downsampling to a target smish ratio inside each split
    val_deploy  = make_deploy_split(val,  target_smish_ratio=DEPLOY_RATIO, seed=SEED)
    test_deploy = make_deploy_split(test, target_smish_ratio=DEPLOY_RATIO, seed=SEED)

    # Write outputs to the processed folder
    PROC.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROC/"train.parquet", index=False)
    val.to_parquet(PROC/"val.parquet", index=False)
    test.to_parquet(PROC/"test.parquet", index=False)
    train_bal.to_parquet(PROC/"train_balanced.parquet", index=False)
    val_deploy.to_parquet(PROC/"val_deploy.parquet", index=False)
    test_deploy.to_parquet(PROC/"test_deploy.parquet", index=False)

    # Compact split summary for quick confirmation
    print("\n=== Split Summary ===")
    for name, d in [
        ("train", train), ("val", val), ("test", test),
        ("train_balanced", train_bal), ("val_deploy", val_deploy), ("test_deploy", test_deploy)
    ]:
        print(summarize(name, d))

if __name__ == "__main__":
    main()
