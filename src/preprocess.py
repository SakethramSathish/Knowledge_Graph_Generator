"""
preprocess.py

Text cleaning and sentence splitting utilities.
"""
import re
from typing import List
import nltk

def _ensure_punkt() -> bool:
    """
    Try to ensure the NLTK 'punkt' tokenizer is available.
    Returns True when punkt is available, False otherwise.
    """
    try:
        nltk.data.find("tokenizers/punkt")
        return True
    except LookupError:
        pass
    try:
        # attempt to download quietly
        nltk.download("punkt", quiet=True)
        nltk.data.find("tokenizers/punkt")
        return True
    except Exception:
        return False

_PUNKT_AVAILABLE = _ensure_punkt()

from nltk.tokenize import sent_tokenize

def clean_text(text:str) -> str:
    #Normalize newlines and whitespace, remove non-printable unicode
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    # remove non-ascii control characters but keep common punctuation
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def split_sentences(text:str) -> List[str]:
    text = clean_text(text)
    if _PUNKT_AVAILABLE:
        try:
            sents = sent_tokenize(text)
        except LookupError:
            # fall back if punkt is somehow missing at runtime
            _ = _ensure_punkt()
            try:
                sents = sent_tokenize(text)
            except Exception:
                sents = _simple_sent_split(text)
    else:
        sents = _simple_sent_split(text)
    return [s.strip() for s in sents if len(s.strip()) > 3]

def _simple_sent_split(text: str) -> List[str]:
    """
    Lightweight fallback sentence splitter when NLTK punkt is unavailable.
    Not as accurate as punkt, but robust for basic text.
    """
    # Split on sentence end punctuation followed by space/newline
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    return [p for p in parts if p.strip()]