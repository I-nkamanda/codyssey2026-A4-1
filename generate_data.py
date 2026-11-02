"""Build a reproducible, synthetic and anonymous set of short movie reviews."""
import csv
import random
from pathlib import Path

rng = random.Random(42)
subjects = ['연기', '스토리', '음악', '영상', '대사', '연출', '결말', '액션', '전개', '인물']
positive = ['좋았다', '인상적이었다', '만족스러웠다', '훌륭했다', '마음에 들었다', '재미있었다']
negative = ['아쉬웠다', '지루했다', '실망스러웠다', '별로였다', '마음에 들지 않았다', '어색했다']
contexts = ['극장에서 봤는데', '기대하고 봤는데', '주말에 봤는데', '끝까지 봤는데', '다시 봤는데', '처음 봤는데']
connectors = ['그리고', '반면', '하지만', '또한']
rows = []
for i in range(1200):
    label = i % 2
    main = positive if label else negative
    other = negative if label else positive
    # Mixed sentiment and occasional inversion make the classification problem nontrivial.
    a, b = rng.sample(subjects, 2)
    first, second = rng.choice(main), rng.choice(other if rng.random() < .70 else main)
    if rng.random() < .45:
        text = f'{rng.choice(contexts)} {a} {first} {rng.choice(connectors)} {b} {second}'
    else:
        text = f'{rng.choice(contexts)} {b} {second} {rng.choice(connectors)} {a} {first}'
    rows.append((f'review-{i+1:04d}', text, label))
out = Path(__file__).resolve().parent / 'data/reviews.csv'
out.parent.mkdir(exist_ok=True)
with out.open('w', encoding='utf-8', newline='') as f:
    w = csv.writer(f); w.writerow(['id', 'text', 'label']); w.writerows(rows)
