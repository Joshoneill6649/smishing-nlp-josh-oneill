#!/usr/bin/env python3
# scripts/normalize_merge_english.py
"""
Normalize + merge local sources into one parquet with a lenient, layered English filter.

Looks for these files (either location works):
- data/raw/sms_spam_collection.csv        or  data/raw/sources/sms_spam_collection.csv
- data/raw/smishtank.csv                  or  data/raw/sources/smishtank.csv
- data/raw/spamdam.csv                    or  data/raw/sources/spamdam.csv
- data/raw/NUS_SMS_Corpus.json            or  data/raw/sources/NUS_SMS_Corpus.json

Outputs:
- data/processed/combined.parquet
- data/processed/non_english_dropped.parquet  (for audit)
- reports/normalize_merge_summary.json
"""

from pathlib import Path
import json, re, math
import pandas as pd
import langid

# ---------- resolve repo root & candidate dirs ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC1 = REPO_ROOT / "data" / "raw" / "sources"
SRC2 = REPO_ROOT / "data" / "raw"
PROC  = REPO_ROOT / "data" / "processed"; PROC.mkdir(parents=True, exist_ok=True)
REPORTS = REPO_ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)

def first_existing(*cands: Path) -> Path | None:
    for p in cands:
        if p.exists():
            return p
    return None

RAW_UCI  = first_existing(SRC1/"sms_spam_collection.csv", SRC2/"sms_spam_collection.csv")
RAW_ST   = first_existing(SRC1/"smishtank.csv",          SRC2/"smishtank.csv")
RAW_SD   = first_existing(SRC1/"spamdam.csv",            SRC2/"spamdam.csv")
RAW_NUSJ = first_existing(SRC1/"NUS_SMS_Corpus.json",    SRC2/"NUS_SMS_Corpus.json")

# ---------- normalization ----------
URL   = re.compile(r'https?://\S+|www\.\S+')
PHONE = re.compile(r'\+?\d[\d\s().-]{6,}\d')
EMAIL = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

def norm(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = URL.sub("<URL>", s)
    s = EMAIL.sub("<EMAIL>", s)
    s = PHONE.sub("<PHONE>", s)
    return " ".join(s.split())

# ---------- lenient English-ish heuristics ----------
STOPWORDS = {
    "the","to","and","you","for","your","on","in","is","of","it","this","that",
    "with","we","are","be","as","at","or","from","by","an","if","not","have",
    "please","now","here","there","will","can"
}

LETTERS = re.compile(r"[a-z]")

def ascii_ratio(s: str) -> float:
    if not s: return 0.0
    a = sum(1 for ch in s if ord(ch) < 128)
    return a / len(s)

def letters_ratio(s: str) -> float:
    if not s: return 0.0
    a = sum(1 for ch in s if ch.isalpha())
    return a / max(1, len(s))

def stopword_hits(s: str) -> int:
    toks = re.findall(r"[a-z]+", s.lower())
    return sum(1 for t in toks if t in STOPWORDS)

def is_en_langid(s: str, p_thresh: float) -> bool:
    code, p = langid.classify(s)
    return (code == "en") and (p >= p_thresh)

def englishish_lenient(s: str) -> bool:
    """Accept short/URL-heavy English messages generously."""
    if not isinstance(s, str): return False
    t = s.strip()
    if len(t) < 2: return False
    # 1) langid with low threshold
    if is_en_langid(t, 0.60):  # lenient
        return True
    # 2) stopwords heuristic
    if stopword_hits(t) >= 2:
        return True
    # 3) ascii+letters heuristic
    if ascii_ratio(t) >= 0.95 and letters_ratio(t) >= 0.35:
        return True
    return False

def englishish_for_nus(s: str) -> bool:
    """Slightly stricter for NUS ham to avoid non-English ham leakage."""
    if not isinstance(s, str): return False
    t = s.strip()
    if len(t) < 2: return False
    if is_en_langid(t, 0.70):  # a bit stricter
        return True
    if stopword_hits(t) >= 3:
        return True
    if ascii_ratio(t) >= 0.97 and letters_ratio(t) >= 0.40:
        return True
    return False

# ---------- loaders ----------
def load_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t", header=None, names=["v1","v2"])

def load_uci(path: Path) -> pd.DataFrame:
    df = load_csv_any(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1":"label_raw","v2":"text"})
    elif "label" in df.columns and "message" in df.columns:
        df = df.rename(columns={"label":"label_raw","message":"text"})
    else:
        text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
        label_col = [c for c in df.columns if c != text_col][0]
        df = df.rename(columns={label_col:"label_raw", text_col:"text"})
    lab = df["label_raw"].astype(str).str.lower().str.strip()
    df["label"] = (lab != "ham").astype(int)  # ham=0, spam/smish=1
    out = df[["text","label"]].copy()
    out["source"] = "uci_sms"
    return out

def load_smish_only_csv(path: Path, source_name: str) -> pd.DataFrame:
    df = load_csv_any(path)
    df.columns = [c.lower().strip() for c in df.columns]
    text_col = next((c for c in ["text","message","sms","content","body","msg"] if c in df.columns), df.columns[0])
    out = df[[text_col]].rename(columns={text_col:"text"})
    out["label"] = 1
    out["source"] = source_name
    return out

def load_nus_json(path: Path) -> pd.DataFrame:
    data = json.load(open(path, "r", encoding="utf-8"))
    root = data.get("smsCorpus", {})
    msgs = root.get("message", [])
    if isinstance(msgs, dict): msgs = [msgs]

    def extract_str(val):
        if isinstance(val, str): return val.strip()
        if isinstance(val, dict):
            if "$" in val and isinstance(val["$"], str):
                return val["$"].strip()
            for v in val.values():
                s = extract_str(v)
                if isinstance(s, str) and s: return s
        if isinstance(val, list):
            for v in val:
                s = extract_str(v)
                if isinstance(s, str) and s: return s
        return None

    rows = []
    for rec in msgs:
        if not isinstance(rec, dict): continue
        txt = None
        if "text" in rec:
            txt = extract_str(rec["text"])
        if not txt:
            for k in ("message","body","content","sms","msg","messageText","message_text"):
                if k in rec:
                    txt = extract_str(rec[k]); 
                    if txt: break
        if txt: rows.append({"text": txt})

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        raise SystemExit("Found 'smsCorpus' but couldn't extract message text from its 'message' entries.")
    df["label"] = 0
    df["source"] = "nus_sms"
    return df[["text","label","source"]]

# ---------- main ----------
def main():
    print("[paths] searching in:")
    print("  ", SRC1)
    print("  ", SRC2)

    parts = []
    if RAW_UCI:  print(f"[load] UCI       : {RAW_UCI}");  parts.append(load_uci(RAW_UCI))
    if RAW_ST:   print(f"[load] SmishTank : {RAW_ST}");   parts.append(load_smish_only_csv(RAW_ST, "smishtank"))
    if RAW_SD:   print(f"[load] SpamDam   : {RAW_SD}");   parts.append(load_smish_only_csv(RAW_SD, "spamdam"))
    if RAW_NUSJ: print(f"[load] NUS JSON  : {RAW_NUSJ}"); parts.append(load_nus_json(RAW_NUSJ))

    if not parts:
        raise SystemExit(
            "No inputs found. Expected any of:\n"
            f"  {SRC1/'sms_spam_collection.csv'}\n  {SRC2/'sms_spam_collection.csv'}\n"
            f"  {SRC1/'smishtank.csv'}\n  {SRC2/'smishtank.csv'}\n"
            f"  {SRC1/'spamdam.csv'}\n  {SRC2/'spamdam.csv'}\n"
            f"  {SRC1/'NUS_SMS_Corpus.json'}\n  {SRC2/'NUS_SMS_Corpus.json'}"
        )

    df = pd.concat(parts, ignore_index=True)
    print(f"[info] Combined (pre-clean): N={len(df):,}")
    print(df["source"].value_counts())

    # normalize
    print("[step] normalize text…")
    df["text"] = df["text"].map(norm)
    df = df[df["text"].str.len() > 0]

    # PER-SOURCE English-ish filter:
    print("[step] english filter (per-source, lenient)…")
    is_uci = df["source"].eq("uci_sms")
    is_nus = df["source"].eq("nus_sms")
    is_other = ~(is_uci | is_nus)

    mask = pd.Series(False, index=df.index)
    mask |= is_uci  # trust UCI English documentation
    mask |= df.loc[is_nus, "text"].map(englishish_for_nus).reindex(df.index, fill_value=False)
    mask |= df.loc[is_other, "text"].map(englishish_lenient).reindex(df.index, fill_value=False)

    before_en = len(df)
    kept = df[mask].copy()
    dropped = df[~mask].copy()

    print(f"[info] kept after EN: {len(kept):,} ({len(kept)/before_en*100:.1f}%)")
    print("[info] kept by source:")
    print(kept["source"].value_counts())

    # dedupe
    print("[step] dedupe on text…")
    before = len(kept)
    kept = kept.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[info] removed dups: {before - len(kept):,}")

    # write outputs
    out_path = PROC / "combined.parquet"
    kept.to_parquet(out_path, index=False)

    # also write dropped for audit
    if not dropped.empty:
        dropped_path = PROC / "non_english_dropped.parquet"
        dropped.to_parquet(dropped_path, index=False)
    else:
        dropped_path = None

    summary = {
        "total_raw": int(len(df)),
        "kept_total": int(len(kept)),
        "dropped_total": int(len(dropped)),
        "ham_kept":   int((kept.label == 0).sum()),
        "smish_kept": int((kept.label == 1).sum()),
        "ham_pct_kept": float((kept.label == 0).mean() * 100.0),
        "kept_by_source": kept["source"].value_counts().to_dict(),
        "dropped_by_source": dropped["source"].value_counts().to_dict(),
        "notes": "Lenient English-ish filter: langid>=0.60 OR ≥2 stopwords OR high ASCII/letters. NUS slightly stricter.",
    }
    with open(REPORTS / "normalize_merge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] Wrote {out_path}")
    if dropped_path:
        print(f"[audit] Dropped (non-English-ish) → {dropped_path}")
    print("Summary → reports/normalize_merge_summary.json")
    print(f"Counts kept: N={summary['kept_total']:,} | ham={summary['ham_kept']:,} ({summary['ham_pct_kept']:.2f}%) | smish={summary['smish_kept']:,}")
    print("Kept by source:", summary["kept_by_source"])

if __name__ == "__main__":
    main()
