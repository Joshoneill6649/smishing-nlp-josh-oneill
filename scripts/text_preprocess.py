import re, unicodedata

#used as an extra cleanup file when needed (predict_message.py)

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def normalize_text(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKC", s)  # fix unicode look-alikes
    s = s.replace("\u200b", "")           # strip zero-width chars
    s = s.lower().strip()                 # Convert to lower case and trim leading and trailing spaces
    s = URL_RE.sub("[URL]", s)            # same token you used in build script
    s = re.sub(r"\s+", " ", s)            # Collapse runs of whitespace into a single space
    return s



