"""NumPy-based TF-IDF vectorizer and cosine similarity calculation."""
from __future__ import annotations

from collections import Counter #Counter 클래서를 가져옴.

import numpy as np #NumPy 라이브러리를 np로 가져옵니다. (np.method() 형식으로 불러옴)

from preprocess import tokenize #preprocess.py 에서 tokenize 함수를 가져옵니다.


class Tfidf: #TF-IDF를 위한 새로운 클래스를 정의합니다. Tfidf 클래스는 fit() 실행 시 self.vocabulary와 self.idf라는 내부 상태를 저장합니다. __init__()이 생략된 이유는 객체를 만들 때(Tfidf()) 미리 넘길 설정값이 없고, fit() 단계에서 속성을 생성하기 때문입니다.
    """TF-IDF vectorizer using raw TF, smoothed IDF, and L2 normalization."""

    def fit(self, documents: list[str]) -> 'Tfidf': #객체와 리뷰 text 데이터를 받아서 Tfidf 객체를 반환한다 (__init__()이 없어도 되는 이유)
        """Build vocabulary and compute smoothed IDF from training documents."""
        tokens = [tokenize(d) for d in documents] #모든 리뷰 텍스트를 토큰화한다.
        self.vocabulary = {word: i for i, word in enumerate(sorted({w for doc in tokens for w in doc}))} 
        # {} -> tokens에 있는 모든 토큰을 하나의 집합으로 모아줌(중복을 제거해 준다). #sorted() -> 집합을 정렬한 list를 반환함. 
        # #enumerate() -> 순회하면서 (0부터)(인덱스, 단어) 쌍을 반환함. 
        # # {word: i for i, word in enumerate(...)} -> 이걸 통해서 self.vocabulary = {'감동': 0, '연기':1, ..} 이런식으로 각 단어를 key, index를 value로 가지는 딕셔너리를 만들어서 저장한다
        # #vocabulary 내의 모든 단어가 숫자 벡터로 표현이 된다.
        counts = self._counts(tokens) #토큰을 _counts라는 메서드에 전달 후 반환 결과를 counts에 저장해 줌.
        self.idf = np.log((1 + len(documents)) / (1 + np.count_nonzero(counts, axis=0))) + 1 #IDF 공식을 그대로 옮겨놓은 것이다.
        #len(documents) = 전체 문서 수 / np.log() = 괄호 아래 부분을 자연로그를 취함 
        #np.count_nonzero(counts, axis=0) = self._counts로 계산한 matrix에서 각 단어가 몇 개 문서에 등장했는지 계산해주는 method. (각 단어의 DF)
        #분자 / 분모의 +1 = smoothing. 마지막 +1 = 모든 단어의 IDF가 최소 1이 되도록 조정
        return self

    def _counts(self, tokenized_docs: list[list[str]]) -> np.ndarray: #토큰화된 문서들의 리스트를 받아서 (문서 수 X 고유 단어 수)의 2 차원 array로 반환을 해준다.
        """Build a (num_docs, vocab_size) term-frequency matrix."""
        matrix = np.zeros((len(tokenized_docs), len(self.vocabulary)), dtype=np.float64) # 총 문서 갯수 X 고유 단어 갯수로 된 float datatype의 0으로 가득찬 행렬을 생성(np.zeros())한다. 각 원소는 64비트 부동소수점 수로 저장한다. 
        for i, words in enumerate(tokenized_docs): # 각 문서에 번호를 붙여서 순회한다
            for word, count in Counter(words).items(): #('영화':2, '재미':1)와 같이 단어를 count한 값을 가져온다.
                if word in self.vocabulary: #단어가 vocabulary에 존재하면
                    matrix[i, self.vocabulary[word]] = count # 행렬의 [문서 인덱스, 단어 인덱스]위치에 출현 횟수를 기록한다.
        return matrix #완성된 단어 출현 횟수 행렬을 반환한다.

    def transform(self, documents: list[str]) -> np.ndarray: #documents를 벡터로 변환한다. L2 normalizations이란: 벡터의 방향은 유지하면서 길이를 1로 만드는 작업.
        #주의: 원소들의 제곱 합의 제곱근을 1로 만든다는 것이다. (피타고라스 정리!)
        """Transform documents into L2-normalized TF-IDF vectors."""
        matrix = self._counts([tokenize(d) for d in documents]) * self.idf #문서를 토큰화한 뒤에, 문서별 단어 출현 횟수 행렬을 만들어 주고, 각 단어의 IDF 값을 곱해서 TF-IDF 행렬을 만든다.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) #각 문서(행)마다 벡터의 길이(L2 Norm)를 계산해 준다. 
        #np.linalg.norm = NumPy.Linear_Algebra_normalization (벡터의 길이를 계산해주며, 기본 설정으로 L2 Norm을 사용함.) axis=1 :각 행을 따라 계산해줌. (문서별로 벡터의 길이를 구해줌) keepdims=True: dimension(2차원 배열)의 형태를 유지한다.
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0) #np.divide() = 원소별 나눗셈.
        # np.divide(나눠질 행렬, 각 문서 벡터의 길이, 계산결과를 저장할 배열 = matrix와 같은 dimension의 0행렬, 길이가 0인 벡터는 나누지 않고 0으로 남긴다-오류 방지용)
# 그럼 정규화를 왜 할까요? -> 문서 A가 문서 B의 몇 배가 길어서 단어가 더 많이 등장했다고 할지라도, 그것만으로 검색어와 의미상 더 관련있다고 볼 수는 없다. 정규화를 통해 벡터의 길이 차이를 제거하면 이제 단어 가중치의 "상대적인 분포와 방향"을 중심으로 비교할 수 있다.
# 또한, 코사인 유사도의 경우도 두 벡터 사이의 각도를 이용하는데, 정규화가 되어서 길이가 1이다? 이러면 내적만 써도 코사인 유사도를 계산할 수 있게 되는 것이다.

def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray: #검색어 벡터와 문서 벡터의 코사인 유사도를 계산해서, 각 문서의 유사도 점수를 ndarray로 반환해 준다.
    """Compute cosine similarity between a query vector and each row of matrix."""
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) #denominator를 정해주는데, 각 문서 벡터의 L2 길이와 검색어 벡터의 L2 길이를 원소별 곱셈을 진행해 줘서 코사인 유사도 공식의 분모를 계산한다.
    #* 는 원소곱, @ 는 내적 기호이다. 
    return np.divide(matrix @ query, denominator, out=np.zeros(len(matrix)), where=denominator != 0) #코사인 유사도 공식대로 matrix와 query의 내적값을 denominator로 나눠서 코사인 유사도를 구해준다. 분모가 0인 경우에는 나눗셈을 하지 않고 0으로 나눠둬서 에러를 방지한다.
