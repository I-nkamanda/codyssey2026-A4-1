"""Document search over review corpus using TF-IDF and cosine similarity."""
from __future__ import annotations

from preprocess import load_data
from tfidf import Tfidf, cosine


def search(query: str, topk: int = 5) -> list[tuple[float, str, str]]:
    """Return up to topk (score, doc_id, text_snippet) results sorted by score desc, id asc."""
    rows = load_data()
    texts = [r['text'] for r in rows]
    vectorizer = Tfidf().fit(texts)
    scores = cosine(vectorizer.transform([query])[0], vectorizer.transform(texts))
    indices = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]['id']))
    return [(float(scores[i]), rows[i]['id'], texts[i][:120]) for i in indices[:topk] if scores[i] > 0]
