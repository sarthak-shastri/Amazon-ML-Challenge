"""Reusable text normalization for business names and addresses.

No geocoding / internet lookups / external databases are used anywhere here —
this module is pure string processing.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

try:
    from unidecode import unidecode as _unidecode
    _HAS_UNIDECODE = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_UNIDECODE = False

# --- Legal-form / suffix normalization (applied to business names) ---------
LEGAL_SUFFIX_MAP = {
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "co": "company",
    "company": "company",
    "llc": "limitedliabilitycompany",
    "llp": "limitedliabilitypartnership",
    "plc": "publiclimitedcompany",
    "gmbh": "gmbh",
    "sa": "sa",
    "sarl": "sarl",
}

# --- Address abbreviation normalization -------------------------------------
ADDRESS_ABBR_MAP = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "apt": "apartment",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "ste": "suite",
    "sq": "square",
    "pl": "place",
    "ct": "court",
    "no": "number",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def _unicode_fold(text: str) -> str:
    """Transliterate to a comparable ASCII-ish form.

    Uses the local (offline, no network) `unidecode` library when available,
    which handles both accented Latin scripts (French, Spanish, ...) and a
    best-effort phonetic transliteration of non-Latin scripts (Devanagari,
    Cyrillic, ...). This is the "optional local transliteration" mentioned in
    the spec -- it is a pure lookup-table transliteration library, not an
    external service. Falls back to plain NFKD ASCII-folding (which drops
    non-Latin characters entirely) if unidecode is not installed.
    """
    if _HAS_UNIDECODE:
        return _unidecode(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def _base_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.lower()
    text = _unicode_fold(text)
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Normalize a business name: casing, unicode, punctuation, legal suffixes."""
    text = _base_clean(name)
    if not text:
        return ""
    tokens = text.split(" ")
    normed_tokens = [LEGAL_SUFFIX_MAP.get(tok, tok) for tok in tokens]
    return " ".join(normed_tokens)


def normalize_address(address: str) -> str:
    """Normalize an address: casing, unicode, punctuation, street abbreviations.

    Numbers (street numbers, postal codes, unit numbers) are preserved as-is.
    """
    text = _base_clean(address)
    if not text:
        return ""
    tokens = text.split(" ")
    normed_tokens = [ADDRESS_ABBR_MAP.get(tok, tok) for tok in tokens]
    return " ".join(normed_tokens)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in text.split(" ") if t]


def numeric_tokens(text: str) -> List[str]:
    """Extract numeric tokens (street numbers, postal codes, unit numbers)."""
    if not text:
        return []
    return _DIGIT_RE.findall(text)


def postal_code_candidates(address_norm: str) -> List[str]:
    """Heuristic postal-code extraction: numeric tokens of length >= 4.

    Works across many country formats (US 5-digit ZIP, Indian 6-digit PIN,
    generic alphanumeric postcodes reduced to their digit runs) without
    hard-coding any specific country's format.
    """
    return [t for t in numeric_tokens(address_norm) if len(t) >= 4]


def street_number_candidate(address_norm: str) -> str:
    """Heuristic leading street-number extraction: first numeric token, if any,
    that appears before the 6th word of the address (keeps it 'near the front')."""
    tokens = tokenize(address_norm)
    for tok in tokens[:6]:
        if tok.isdigit():
            return tok
    return ""
