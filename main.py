"""Anonymous synthetic review search and TF-IDF experiment."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STOPWORDS = {'이', '그', '저', '것', '수', '정말', '너무', '그냥', '그리고', '하지만', '영화'}
TOKEN = re.compile(r'[가-힣a-z0-9]+')


def tokenize(text):
    return [w for w in TOKEN.findall(text.lower()) if w not in STOPWORDS]


class Tfidf:
    def fit(self, documents):
        tokens = [tokenize(d) for d in documents]
        self.vocabulary = {word: i for i, word in enumerate(sorted({w for doc in tokens for w in doc}))}
        counts = self._counts(tokens)
        self.idf = np.log((1 + len(documents)) / (1 + np.count_nonzero(counts, axis=0))) + 1
        return self

    def _counts(self, documents):
        matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=np.float64)
        for i, words in enumerate(documents):
            for word, count in Counter(words).items():
                if word in self.vocabulary:
                    matrix[i, self.vocabulary[word]] = count
        return matrix

    def transform(self, documents):
        matrix = self._counts([tokenize(d) for d in documents]) * self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def cosine(query, matrix):
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.divide(matrix @ query, denominator, out=np.zeros(len(matrix)), where=denominator != 0)


def load_data():
    with (ROOT / 'data/reviews.csv').open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def search(query, topk=5):
    rows = load_data()
    texts = [r['text'] for r in rows]
    vectorizer = Tfidf().fit(texts)
    scores = cosine(vectorizer.transform([query])[0], vectorizer.transform(texts))
    indices = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]['id']))
    return [(float(scores[i]), rows[i]['id'], texts[i][:120]) for i in indices[:topk] if scores[i] > 0]


def experiment():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = load_data()
    texts = [r['text'] for r in rows]
    labels = np.array([int(r['label']) for r in rows])
    train, test = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=42, stratify=labels)
    tfidf = Tfidf().fit([texts[i] for i in train])
    x_train = tfidf.transform([texts[i] for i in train])
    x_test = tfidf.transform([texts[i] for i in test])
    reference = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                vocabulary=tfidf.vocabulary, smooth_idf=True,
                                sublinear_tf=False, norm='l2', dtype=np.float64)
    ref_train = reference.fit_transform([texts[i] for i in train]).toarray()
    ref_test = reference.transform([texts[i] for i in test]).toarray()
    diffs = np.concatenate([np.abs(x_train - ref_train).ravel(), np.abs(x_test - ref_test).ravel()])
    model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear').fit(x_train, labels[train])
    pred = model.predict(x_test)
    cm = confusion_matrix(labels[test], pred, labels=[0, 1])
    result = {'documents': len(rows), 'train': len(train), 'test': len(test), 'features': len(tfidf.vocabulary),
              'max_error': float(diffs.max()), 'mean_error': float(diffs.mean()),
              'accuracy': float(accuracy_score(labels[test], pred)),
              'macro_f1': float(f1_score(labels[test], pred, average='macro')), 'confusion_matrix': cm.tolist(),
              'misclassified': [{'id': rows[i]['id'], 'text': texts[i], 'actual': int(labels[i]), 'predicted': int(p)}
                                for i, p in zip(test, pred) if labels[i] != p]}
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'], xlabel='Predicted', ylabel='Actual', title='Confusion matrix')
    fig.tight_layout()
    fig.savefig(ROOT / 'results/confusion_matrix.png', dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != 'misclassified'}, ensure_ascii=False, indent=2))
    assert result['max_error'] <= 1e-6, 'TF-IDF verification failed'


def main():
    parser = argparse.ArgumentParser(description='TF-IDF review search')
    parser.add_argument('--query', default='연기 감동')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--experiment', action='store_true')
    args = parser.parse_args()
    if args.topk < 1:
        parser.error('--topk must be positive')
    if args.experiment:
        experiment()
    else:
        results = search(args.query, args.topk)
        if not results:
            print('일치하는 문서가 없습니다.')
        for score, doc_id, snippet in results:
            print(f'[{score:.4f}] {doc_id} {snippet}')


if __name__ == '__main__':
    main()
