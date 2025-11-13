#!/usr/bin/env python3
# scripts/normalize_merge_english.py
# Merge local SMS sources into one DataFrame, normalize text, keep rows that look English,
# drop exact duplicate texts, then write a combined parquet and a summary JSON.
# Also writes an audit parquet of rows dropped by the English filter when any are dropped.
#
# INPUTS
# sms_spam_collection.csv in data/raw or data/raw/sources
# smishtank.csv in data/raw or data/raw/sources
# spamdam.csv in data/raw or data/raw/sources
# NUS_SMS_Corpus.json in data/raw or data/raw/sources
#
# OUTPUTS
# data/processed/combined.parquet
# data/processed/non_english_dropped.parquet only if something was dropped
# reports/normalize_merge_summary.json

from pathlib import Path
import json
import re
from typing import Optional

import pandas as pd
import langid

# PATHS AND FOLDERS
# Anchor paths at the repo root so behavior is stable from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC1 = REPO_ROOT / "data" / "raw" / "sources"
SRC2 = REPO_ROOT / "data" / "raw"

# Ensure output folders exist before writing.
PROC = REPO_ROOT / "data" / "processed"; PROC.mkdir(parents=True, exist_ok=True)
REPORTS = REPO_ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)

# FILE DISCOVERY
# Return the first existing path from candidates to support two raw layouts.
def first_existing(*cands: Path) -> Optional[Path]:
    for p in cands:
        if p.exists():
            return p
    return None

# Probe for inputs in either location.
RAW_UCI  = first_existing(SRC1/"sms_spam_collection.csv", SRC2/"sms_spam_collection.csv")
RAW_ST   = first_existing(SRC1/"smishtank.csv",          SRC2/"smishtank.csv")
RAW_SD   = first_existing(SRC1/"spamdam.csv",            SRC2/"spamdam.csv")
RAW_NUSJ = first_existing(SRC1/"NUS_SMS_Corpus.json",    SRC2/"NUS_SMS_Corpus.json")

# NORMALIZATION
# Mask URLs emails and phone numbers with fixed tokens and tidy the rest.
URL   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PHONE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
EMAIL = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")

# Normalize one message. Trim, lowercase, replace tokens, collapse spaces.
def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = URL.sub("<URL>", s)
    s = EMAIL.sub("<EMAIL>", s)
    s = PHONE.sub("<PHONE>", s)
    return " ".join(s.split())

# ENGLISHISH HEURISTICS
# Small stopword list to preserve short English messages that langid may under score.
STOPWORDS = {
    "the","to","and","you","for","your","on","in","is","of","it","this","that",
    "with","we","are","be","as","at","or","from","by","an","if","not","have",
    "please","now","here","there","will","can"
}

# Share of ASCII characters. High values often indicate English like text.
def ascii_ratio(s: str) -> float:
    return 0.0 if not s else sum(1 for ch in s if ord(ch) < 128) / len(s)

# Share of alphabetic characters. Filters symbol heavy or shortcode noise.
def letters_ratio(s: str) -> float:
    return 0.0 if not s else sum(1 for ch in s if ch.isalpha()) / max(1, len(s))

# Count how many common English words appear.
def stopword_hits(s: str) -> int:
    return sum(1 for t in re.findall(r"[a-z]+", s.lower()) if t in STOPWORDS)

# langid gate. Accept only if predicted English with probability at or above threshold.
def is_en_langid(s: str, p_thresh: float) -> bool:
    code, p = langid.classify(s)
    return (code == "en") and (p >= p_thresh)

# General lenient Englishish check
# Accept if langid >= 0.60 or stopwords >= 2 or ASCII >= 0.95 and letters >= 0.35.
def englishish_lenient(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip()
    if len(t) < 2:
        return False
    if is_en_langid(t, 0.60):
        return True
    if stopword_hits(t) >= 2:
        return True
    if ascii_ratio(t) >= 0.95 and letters_ratio(t) >= 0.35:
        return True
    return False

# NUS ham uses a stricter rule to avoid non English rows
# Accept if langid >= 0.70 or stopwords >= 3 or ASCII >= 0.97 and letters >= 0.40.
def englishish_for_nus(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip()
    if len(t) < 2:
        return False
    if is_en_langid(t, 0.70):
        return True
    if stopword_hits(t) >= 3:
        return True
    if ascii_ratio(t) >= 0.97 and letters_ratio(t) >= 0.40:
        return True
    return False

# LOADERS to text label source
# Robust CSV reader. Try normal CSV then TSV with no header which is common for UCI.
def load_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t", header=None, names=["v1","v2"])

# UCI loader. Standardize columns. ham to 0, not ham to 1, then tag the source.
def load_uci(path: Path) -> pd.DataFrame:
    df = load_csv_any(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1":"label_raw","v2":"text"})
    elif "label" in df.columns and "message" in df.columns:
        df = df.rename(columns={"label":"label_raw","message":"text"})
    else:
        # If header is unusual, guess text column by average string length.
        text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
        label_col = [c for c in df.columns if c != text_col][0]
        df = df.rename(columns={label_col:"label_raw", text_col:"text"})
    lab = df["label_raw"].astype(str).str.lower().str.strip()
    df["label"] = (lab != "ham").astype(int)
    out = df[["text","label"]].copy()
    out["source"] = "uci_sms"
    return out

# Smish only CSVs such as SmishTank and SpamDam. Pick a sensible text column, set label 1, tag the source.
def load_smish_only_csv(path: Path, source_name: str) -> pd.DataFrame:
    df = load_csv_any(path)
    df.columns = [c.lower().strip() for c in df.columns]
    text_col = next((c for c in ["text","message","sms","content","body","msg"] if c in df.columns), df.columns[0])
    out = df[[text_col]].rename(columns={text_col:"text"})
    out["label"] = 1
    out["source"] = source_name
    return out

# NUS JSON loader. Walk a nested shape and extract message text. Set label 0 and tag the source.
def load_nus_json(path: Path) -> pd.DataFrame:
    data = json.load(open(path, "r", encoding="utf-8"))
    root = data.get("smsCorpus", {})
    msgs = root.get("message", [])
    if isinstance(msgs, dict):
        msgs = [msgs]

    # Return a trimmed string from nested string dict or list, else None.
    def extract_str(val):
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, dict):
            if "$" in val and isinstance(val["$"], str):
                return val["$"].strip()
            for v in val.values():
                s = extract_str(v)
                if isinstance(s, str) and s:
                    return s
        if isinstance(val, list):
            for v in val:
                s = extract_str(v)
                if isinstance(s, str) and s:
                    return s
        return None

    rows = []
    for rec in msgs:
        if not isinstance(rec, dict):
            continue
        txt = extract_str(rec.get("text"))
        if not txt:
            # Try other likely keys used in the wild.
            for k in ("message","body","content","sms","msg","messageText","message_text"):
                if k in rec:
                    txt = extract_str(rec[k])
                    if txt:
                        break
        if txt:
            rows.append({"text": txt})

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        raise SystemExit("Found smsCorpus but could not extract message text.")
    df["label"] = 0
    df["source"] = "nus_sms"
    return df[["text","label","source"]]

# MAIN PIPELINE
def main() -> None:
    # Step 1. Load any available sources. It is acceptable if some are missing.
    parts = []
    if RAW_UCI:  parts.append(load_uci(RAW_UCI))
    if RAW_ST:   parts.append(load_smish_only_csv(RAW_ST, "smishtank"))
    if RAW_SD:   parts.append(load_smish_only_csv(RAW_SD, "spamdam"))
    if RAW_NUSJ: parts.append(load_nus_json(RAW_NUSJ))

    # If no sources were found, exit with a clear message that lists the expected paths.
    if not parts:
        raise SystemExit(
            "No inputs found. Expected any of:\n"
            f"  {SRC1/'sms_spam_collection.csv'}\n  {SRC2/'sms_spam_collection.csv'}\n"
            f"  {SRC1/'smishtank.csv'}\n  {SRC2/'smishtank.csv'}\n"
            f"  {SRC1/'spamdam.csv'}\n  {SRC2/'spamdam.csv'}\n"
            f"  {SRC1/'NUS_SMS_Corpus.json'}\n  {SRC2/'NUS_SMS_Corpus.json'}"
        )

    # Step 2. Stack all rows together.
    df = pd.concat(parts, ignore_index=True)

    # Step 3. Normalize text and drop rows that end up empty after masking and trim.
    df["text"] = df["text"].map(norm)
    df = df[df["text"].str.len() > 0]

    # Step 4. Apply the per source Englishish filter
    #         UCI trusted as English
    #         NUS uses stricter gate
    #         Others use lenient gate
    is_uci = df["source"].eq("uci_sms")
    is_nus = df["source"].eq("nus_sms")
    is_other = ~(is_uci | is_nus)

    mask = pd.Series(False, index=df.index)
    mask |= is_uci
    mask |= df.loc[is_nus, "text"].map(englishish_for_nus).reindex(df.index, fill_value=False)
    mask |= df.loc[is_other, "text"].map(englishish_lenient).reindex(df.index, fill_value=False)

    kept = df[mask].copy()
    dropped = df[~mask].copy()

    # Step 5. Drop exact duplicate texts to prevent training bias and inflated counts.
    kept = kept.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Step 6. Write outputs with identical names and schema.
    out_path = PROC / "combined.parquet"
    kept.to_parquet(out_path, index=False)

    dropped_path = None
    if not dropped.empty:
        dropped_path = PROC / "non_english_dropped.parquet"
        dropped.to_parquet(dropped_path, index=False)

    # Step 7. Write summary JSON to capture exact counts for reports and notebooks.
    summary = {
        "total_raw": int(len(df)),
        "kept_total": int(len(kept)),
        "dropped_total": int(len(dropped)),
        "ham_kept":   int((kept.label == 0).sum()),
        "smish_kept": int((kept.label == 1).sum()),
        "ham_pct_kept": float((kept.label == 0).mean() * 100.0),
        "kept_by_source": kept["source"].value_counts().to_dict(),
        "dropped_by_source": dropped["source"].value_counts().to_dict(),
        "notes": "Lenient English style filter with langid or stopword signal or ascii and letters. NUS is a bit stricter.",
    }
    with open(REPORTS / "normalize_merge_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Minimal status to confirm outputs.
    print(f"[done] {out_path}")
    if dropped_path:
        print(f"[audit] dropped → {dropped_path}")
    print("summary → reports/normalize_merge_summary.json")

if __name__ == "__main__":
    main()
