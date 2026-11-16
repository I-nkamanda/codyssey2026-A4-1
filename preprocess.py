"""Text preprocessing, tokenization, and CSV dataset loading."""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STOPWORDS = {'이', '그', '저', '것', '수', '정말', '너무', '그냥', '그리고', '하지만', '영화'}
TOKEN = re.compile(r'[가-힣a-z0-9]+')


def tokenize(text: str) -> list[str]:
    """Lowercase text, extract alphanumeric/Hangul tokens, and remove stopwords."""
    return [w for w in TOKEN.findall(text.lower()) if w not in STOPWORDS]


def load_data(path: Path | None = None) -> list[dict[str, str]]:
    """Load review rows from a UTF-8 CSV file with id, text, and label columns."""
    target = path or (ROOT / 'data/reviews.csv')
    with target.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))
