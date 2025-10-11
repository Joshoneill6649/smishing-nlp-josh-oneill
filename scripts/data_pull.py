#!/usr/bin/env python3
"""
data_pull.py — Fetch raw datasets into data/raw/ (no processing).

- Reads a JSON manifest (URLs or local file paths).
- Downloads to data/raw/sources/<key>/ and extracts archives.
- Copies each dataset's main file to canonical paths:
    data/raw/sms_spam_collection.csv   (UCI, ham+spam)
    data/raw/smishtank.csv             (SmishTank, smish-only)
    data/raw/spamdam.csv               (SpamDam, smish-only)
    data/raw/nus_sms.csv               (NUS ham)

- Writes reports/fetch_report.json with statuses and file sizes.

Run:
    python scripts/data_pull.py --manifest scripts/datasets_manifest.json
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
import tarfile
import zipfile
import gzip
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request

RAW_DIR     = Path("data/raw")
SRC_DIR     = RAW_DIR / "sources"
REPORTS_DIR = Path("reports")
for p in (RAW_DIR, SRC_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_MANIFEST = Path("scripts/datasets_manifest.json")

# Canonical outputs used by later scripts
CANONICAL_TARGETS = {
    "uci":       RAW_DIR / "sms_spam_collection.csv",
    "smishtank": RAW_DIR / "smishtank.csv",
    "spamdam":   RAW_DIR / "spamdam.csv",
    "nus":       RAW_DIR / "nus_sms.csv",
}

def http_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dest, "wb") as f:
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)

def is_archive(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".gz"))

def extract_archive(src: Path, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    n = src.name.lower()
    if n.endswith(".zip"):
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(outdir)
            extracted = [outdir / m for m in z.namelist()]
    elif n.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz")):
        with tarfile.open(src, "r:*") as t:
            t.extractall(outdir)
            extracted = [outdir / m in t.getnames()]
            extracted = [outdir / m for m in t.getnames()]
    elif n.endswith(".gz") and not n.endswith((".tar.gz", ".tgz")):
        out_path = outdir / src.with_suffix("").name
        with gzip.open(src, "rb") as g, open(out_path, "wb") as f:
            shutil.copyfileobj(g, f)
        extracted = [out_path]
    else:
        extracted = [src]
    return extracted

def find_first_csv(folder: Path) -> Path | None:
    csvs = sorted(folder.rglob("*.csv"))
    if csvs:
        return csvs[0]
    tsvs = sorted(folder.rglob("*.tsv"))
    if tsvs:
        return tsvs[0]
    txts = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".data"}])
    return txts[0] if txts else None

def copy_canonical(source_key: str, src_folder: Path, explicit_file: str | None, target: Path) -> Path | None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if explicit_file:
        cand = (src_folder / explicit_file)
        if cand.exists():
            shutil.copy2(cand, target)
            return target
    found = find_first_csv(src_folder)
    if found:
        shutil.copy2(found, target)
        return target
    return None

def load_manifest(path: Path) -> dict:
    if not path.exists():
        # Create a starter manifest to edit, then exit
        starter = {
            "uci": {
                "note": "UCI SMS Spam Collection (ham+spam).",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
                "file": "SMSSpamCollection"
            },
            "smishtank": {
                "note": "SmishTank CSV (positives). Provide a direct CSV url or local path.",
                "url": "", "path": ""
            },
            "spamdam": {
                "note": "SpamDam CSV (positives). Provide a direct CSV url or local path.",
                "url": "", "path": ""
            },
            "nus": {
                "note": "NUS SMS Corpus (ham). Usually manual download due to license.",
                "url": "", "path": ""
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(starter, f, indent=2)
        print(f"[info] Created starter manifest at {path}. Fill it in and re-run.")
        sys.exit(0)
    with open(path) as f:
        return json.load(f)

def fetch_one(key: str, spec: dict) -> dict:
    """
    spec:
      - url: optional HTTP/HTTPS URL
      - path: optional local path already downloaded
      - file: optional preferred file name inside extracted dir to copy as canonical
    """
    dest_root = SRC_DIR / key
    dest_root.mkdir(parents=True, exist_ok=True)

    local_path = spec.get("path") or ""
    url = spec.get("url") or ""
    chosen_dir: Path | None = None
    downloaded_archive: Path | None = None

    if local_path:
        lp = Path(local_path).expanduser().resolve()
        if not lp.exists():
            return {"key": key, "status": "error", "message": f"Local path not found: {lp}"}
        if lp.is_file():
            tmp = dest_root / lp.name
            if not tmp.exists():
                shutil.copy2(lp, tmp)
            if is_archive(tmp):
                extract_archive(tmp, dest_root)
                chosen_dir = dest_root
            else:
                chosen_dir = dest_root
        else:
            chosen_dir = lp
    elif url:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return {"key": key, "status": "error", "message": f"Unsupported URL: {url}"}
        filename = Path(parsed.path).name or f"{key}.download"
        archive_path = dest_root / filename
        print(f"[{key}] downloading: {url}")
        try:
            http_download(url, archive_path)
        except Exception as e:
            return {"key": key, "status": "error", "message": f"Download failed: {e}"}
        downloaded_archive = archive_path
        if is_archive(archive_path):
            extract_archive(archive_path, dest_root)
        chosen_dir = dest_root
    else:
        return {"key": key, "status": "skipped", "message": "No 'url' or 'path' provided."}

    target = CANONICAL_TARGETS.get(key)
    copied = copy_canonical(key, chosen_dir, spec.get("file"), target) if target else None
    size_bytes = target.stat().st_size if (target and target.exists()) else 0
    return {
        "key": key,
        "status": "ok" if (copied or chosen_dir) else "warning",
        "chosen_dir": str(chosen_dir) if chosen_dir else None,
        "canonical": str(target) if target else None,
        "canonical_exists": bool(target and target.exists()),
        "canonical_size_bytes": int(size_bytes),
        "downloaded_archive": str(downloaded_archive) if downloaded_archive else None,
        "message": "" if (copied or chosen_dir) else "No suitable CSV/TXT found."
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to datasets_manifest.json")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    results = {}
    for key in ["uci", "smishtank", "spamdam", "nus"]:
        res = fetch_one(key, manifest.get(key, {}))
        results[key] = res
        msg = f" — {res.get('message','')}" if res.get("message") else ""
        print(f"[{key}] {res['status']}{msg}")

    report = REPORTS_DIR / "fetch_report.json"
    with open(report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[info] Wrote {report}")


if __name__ == "__main__":
    main()



------------ go back to here as I had to drag in datasets manually just test that theory out download them all and add them in -----