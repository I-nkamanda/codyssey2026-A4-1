"""TF-IDF verification against scikit-learn and sentiment classification experiment."""
import json

import numpy as np

from preprocess import ROOT, load_data, tokenize
from tfidf import Tfidf


def experiment() -> None:
    """Verify NumPy TF-IDF against scikit-learn and evaluate Logistic Regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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

    reference = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        vocabulary=tfidf.vocabulary,
        smooth_idf=True,
        sublinear_tf=False,
        norm='l2',
        dtype=np.float64,
    )
    ref_train = reference.fit_transform([texts[i] for i in train]).toarray()
    ref_test = reference.transform([texts[i] for i in test]).toarray()
    diffs = np.concatenate([np.abs(x_train - ref_train).ravel(), np.abs(x_test - ref_test).ravel()])

    model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear').fit(x_train, labels[train])
    pred = model.predict(x_test)
    cm = confusion_matrix(labels[test], pred, labels=[0, 1])

    result = {
        'documents': len(rows),
        'train': len(train),
        'test': len(test),
        'features': len(tfidf.vocabulary),
        'max_error': float(diffs.max()),
        'mean_error': float(diffs.mean()),
        'accuracy': float(accuracy_score(labels[test], pred)),
        'macro_f1': float(f1_score(labels[test], pred, average='macro')),
        'confusion_matrix': cm.tolist(),
        'misclassified': [
            {'id': rows[i]['id'], 'text': texts[i], 'actual': int(labels[i]), 'predicted': int(p)}
            for i, p in zip(test, pred)
            if labels[i] != p
        ],
    }

    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center')
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=['negative', 'positive'],
        yticklabels=['negative', 'positive'],
        xlabel='Predicted',
        ylabel='Actual',
        title='Confusion matrix',
    )
    fig.tight_layout()
    fig.savefig(ROOT / 'results/confusion_matrix.png', dpi=150)
    plt.close(fig)

    print(json.dumps({k: v for k, v in result.items() if k != 'misclassified'}, ensure_ascii=False, indent=2))
    assert result['max_error'] <= 1e-6, 'TF-IDF verification failed'
