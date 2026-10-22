"""Pre-processing from the paper: remove URLs, emojis, hashtags and unwanted characters."""
from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
# keep letters (incl. Spanish accents and ñ), digits and spaces; drop punctuation and emojis
UNWANTED_RE = re.compile(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s]")
WS_RE = re.compile(r"\s+")


def clean_text(text: str, keep_mentions: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    if not keep_mentions:
        text = MENTION_RE.sub(" ", text)
    text = UNWANTED_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()
