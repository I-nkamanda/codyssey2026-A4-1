"""Sample a balanced, cleaned subset of NSMC reviews into data/reviews.csv."""
import csv
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'data/nsmc_ratings_test.txt'
OUT = ROOT / 'data/reviews.csv'

PII_PATTERN = re.compile(r'[\w.-]+@[\w.-]+|\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}')
TOKEN_PATTERN = re.compile(r'[가-힣a-z0-9]+')


def is_valid_review(text: str) -> bool:
    if not text or not text.strip():
        return False
    if PII_PATTERN.search(text):
        return False
    tokens = TOKEN_PATTERN.findall(text.lower())
    return len(tokens) >= 2


def main(n_per_class: int = 1000, seed: int = 42) -> None:
    rng = random.Random(seed)
    pos, neg = [], []
    with SRC.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            doc = (row.get('document') or '').strip()
            label = row.get('label')
            if label not in {'0', '1'} or not is_valid_review(doc):
                continue
            item = (row['id'], doc, int(label))
            (pos if label == '1' else neg).append(item)

    rng.shuffle(pos)
    rng.shuffle(neg)
    sampled = pos[:n_per_class] + neg[:n_per_class]
    rng.shuffle(sampled)

    with OUT.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label'])
        writer.writerows(sampled)

    print(f'Saved {len(sampled)} reviews ({n_per_class} pos / {n_per_class} neg) to {OUT}')


if __name__ == '__main__':
    main()
