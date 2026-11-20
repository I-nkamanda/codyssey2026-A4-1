"""Document search over review corpus using TF-IDF and cosine similarity."""
from __future__ import annotations #Type hint 집행을 미룬다

from preprocess import load_data #preprocess.py 에서 load_data 함수를 불러온다.
from tfidf import Tfidf, cosine #tfidf.py에서 Tfidf 클래스와 cosine 메소드를 불러온다


def search(query: str, topk: int = 5) -> list[tuple[float, str, str]]: #유사도 검사를 위한 search 함수. 문자열 query 및 topk(기본값 5)를 받아서 tuple로 된 list를 반환합니다.
    """Return up to topk (score, doc_id, text_snippet) results sorted by score desc, id asc."""
    rows = load_data() #데이터를 불러와서 rows에 저장합니다
    texts = [r['text'] for r in rows] #rows의 각 행(row)에서 'text'열의 내용을 저장
    vectorizer = Tfidf().fit(texts) #vectorizer라는 TF-IDF 객체를 생성해서 전체 문서에서 단어 사전 과 IDF 값 등 벡터화에 필요한 정보를 학습한다.
    scores = cosine(vectorizer.transform([query])[0], vectorizer.transform(texts)) #query 문구를 벡터화하고, 각 리뷰 문구들도 벡터화한 뒤에, 코사인 유사도 점수를 계산해서 scores에 저장한다.
    indices = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]['id'])) #코사인 유사도를 기준으로 (극성을 바꾼)점수 순서로 오름차순으로 재배열해서 새로운 리스트 indices 로 저장. rows[i]['id'] 는 유사도 점수가 같은 항들의 경우 id의 오름차순으로 정렬한다는 이야기.
    return [(float(scores[i]), rows[i]['id'], texts[i][:120]) for i in indices[:topk] if scores[i] > 0] # 상위 topk(기본 5)개 유사도 점수(0보다 큰 결과만 float로 변환함), id, text snippet(본문의 앞에서부터 최대 120자)을 tuple로 묶어서 반환함
