"""TF-IDF verification against scikit-learn and sentiment classification experiment."""
import json

import numpy as np

from preprocess import ROOT, load_data, tokenize #preprocess.py에서 루트 디렉토리, load 함수 및 tokenize 함수를 불러온다.
from tfidf import Tfidf #Tfidf 객체 및 메소드도 불러온다


def experiment() -> None:
    """Verify NumPy TF-IDF against scikit-learn and evaluate Logistic Regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer #이제 라이브러리에서 그냥 불러오다니 이렇게 편할 데가
    from sklearn.linear_model import LogisticRegression #기본선으로 Logistic Regression 선택함
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score #정확도, 혼동 행렬 및 F1-Score를 가져옴
    from sklearn.model_selection import train_test_split #train/test를 split시켜주는 모듈 import
    import matplotlib #MATLAB을 참고한 시각화 라이브러리 Matplotlib을 가져옵니다.
    matplotlib.use('Agg') #화면에 창을 띄우지 않고 바로 이미지 파일만 생성해 주는 설정.
    #이후 savefig() 등으로 그래프를 이미지 파일에 저장할 수 있다.
    import matplotlib.pyplot as plt #MATLAB 같은 plot을 파이썬에서 그리게 해줍니다

    rows = load_data() #데이터를 로딩합니다
    texts = [r['text'] for r in rows] #text 항만 따로 저장해 줍니다
    labels = np.array([int(r['label']) for r in rows]) #긍정/부정 값만 따로 저장해줍니다.
    train, test = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=42, stratify=labels) #훈련과 테스트를 8:2로 split해줍니다.
    #stratify=labels ->긍정/부정 클래스의 비율을 훈련 및 테스트 set에서 최대한 유지시켜줌
    tfidf = Tfidf().fit([texts[i] for i in train]) #text를 가지고 학습된 Tfidf 객체를 만들어줍니다. fit()을 훈련 데이터에서만 실행 -> 테스트 데이터의 통계가 훈련에 섞이는 data leakage를 방지해 준다.
    x_train = tfidf.transform([texts[i] for i in train]) #train용으로 transform 벡터를 만들어 줍니다
    x_test = tfidf.transform([texts[i] for i in test]) #test 문서를 학습된 TF-IDF 기준으로 벡터화해줌

    reference = TfidfVectorizer( #라이브러리를 호출해서 바로 레퍼런스 객체 및 값을 설정
        tokenizer=tokenize, #동일한 토큰화 함수
        token_pattern=None,
        lowercase=False,
        vocabulary=tfidf.vocabulary, #직접 구현한 TF-IDF와 동일한 단어-인덱스 매핑 사용
        smooth_idf=True, #idf smooting(분자와 분모에 1 더해주는 기법) 적용
        sublinear_tf=False, #로그 변환 없이 원시 출현 횟수 사용
        norm='l2', #동일한 L2 정규화
        dtype=np.float64, #동일한 부동소숫점 자료형
    ) #이렇게 깔끔하게 세팅이 가능하다니, 삽질이 헛된것 같아 눈물이 앞을 가립니다.
    ref_train = reference.fit_transform([texts[i] for i in train]).toarray() #레퍼런스도 트레이닝을 해 줍니다
    ref_test = reference.transform([texts[i] for i in test]).toarray() # 테스트용 레퍼런스 값도 transform을 해 줍니다.
    diffs = np.concatenate([np.abs(x_train - ref_train).ravel(), np.abs(x_test - ref_test).ravel()])
    # x_train - ref_train: 두 구현의 훈련데이터 TF-IDF 값 차이
    # x_test - ref_test: 테스트 데이터 TF-IDF 값 차이
    # np.abs() 괄호 안 계산 결과의 절대값 
    # np.concatenate: 훈련/테스트 데이터의 차이들을 하나의 배열로 합치기
    # .ravel() -> 2차원 배열을 1차원으로 펼쳐줌

    model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear').fit(x_train, labels[train]) #최적화 알고리즘 최대1000회, 난수 시드 42, 최적화 알고리즘(solver)='liblinear, .fit() -> 훈련용 TF-IDF 벡터와 정답 레이블로 모델 학습
    # 훈련용 TF-IDF 벡터(x_train)와 실제 레이블(labels[train])을 사용해
    # Logistic Regression 이진 분류 모델을 학습한다.
    pred = model.predict(x_test) #학습된 모델을 이용, 처음 보는 테스트 데이터(x_test)를 넣고 긍정/부정을 예측함.
    cm = confusion_matrix(labels[test], pred, labels=[0, 1]) #모델이 어떤 종류의 오류를 얼마나 냈는지 혼동 행렬로 만들어주는 함수. 
    # 실제 클래스(행) X 예측 클래스(열)의 혼동 행렬을 생성한다.

    result = { #결과의 자료형은 딕셔너리로 가져간다!
        'documents': len(rows), #몇 개의 리뷰 문서가 있는지
        'train': len(train), #이 중에 훈련용으로는 몇 개를 썼고
        'test': len(test), #테스트용으로는 몇 개를 썼으며
        'features': len(tfidf.vocabulary), #단어(feature) 갯수는 얼마나 되고
        'max_error': float(diffs.max()), #최대 오차(TF-IDF와 scikit-learn의)
        'mean_error': float(diffs.mean()), #평균 오차 (두 구현 사이의)
        'accuracy': float(accuracy_score(labels[test], pred)), #test레이블 Vs 예측치를 가지고 맞힌 예측 수 / 전체 테스트 문서 수로 정확도를 계산하고
        'macro_f1': float(f1_score(labels[test], pred, average='macro')), #깡 정확도로는 클래스별 성능 차이가 가려지기 때문에 정밀도와 재현율의 조화평균인 F1-score를 추가로 사용함. 
        #average = 'macro' ->부정 클래스의 F1과 긍정 클래스의 F1을 각각 구한 뒤에 단순 평균을 계산한다는 이야기. 두 클래스에 동일한 비즁을 부여한다는 것이다.
        'confusion_matrix': cm.tolist(), #NumPy 혼동 행렬을 JSON 저장이 가능한 Python 리스트로 변환해줌.
        'misclassified': [ #틀리게 분류한 리뷰들만 골라서 저장하는 내용의 list comprehension.
            {'id': rows[i]['id'], 'text': texts[i], 'actual': int(labels[i]), 'predicted': int(p)} #id, 본문, 실제 분류, 예측 분류를 담아낸다.
            for i, p in zip(test, pred)
            if labels[i] != p #오분류된 경우에 말이지.
        ],
    }

    (ROOT / 'results').mkdir(exist_ok=True) #결과물을 ROOT/results에 저장한다.
    (ROOT / 'results/metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8') #metrics.json이라는 파일에 위에서 만든 딕셔너리, result를 적어준다.

    fig, ax = plt.subplots(figsize=(5, 4)) #그림 전체, 그래프 축을 5in x 4in으로 그려준다.
    ax.imshow(cm, cmap='Blues') # 축은 혼동행렬(cm으로 상술)을 색상 이미지로 표현한다. 색상은 Blue로, cm의 숫자가 클수록 더 진한 파란색으로 나타남
    for (i, j), val in np.ndenumerate(cm): #NumPy 배열의 각 원소를 index와 함께 순회함
        ax.text(j, i, str(val), ha='center', va='center') #각 cell의 가운데에 숫자를 그려넣는다. ha, va = horizontal / vertical align. 둘 다 center에 위치.
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=['negative', 'positive'],
        yticklabels=['negative', 'positive'],
        xlabel='Predicted', #가로축은 모델이 예측한 클래스
        ylabel='Actual', #세로축은 실제 정답 클래스.
        title='Confusion matrix', #이름도 멋들어지게 붙여준다.
    )
    fig.tight_layout() #제목/축 이름등이 겹치지 않게 여백 조정
    fig.savefig(ROOT / 'results/confusion_matrix.png', dpi=150) # 150dpi로 png 이미지 저장
    plt.close(fig) #그림 객체 닫기

    print(json.dumps({k: v for k, v in result.items() if k != 'misclassified'}, ensure_ascii=False, indent=2)) #result에서 misclassified 항목만 제외한 새 dictionary를 만들어준다.
    assert result['max_error'] <= 1e-6, 'TF-IDF verification failed' #max_error가 1e-6보다 크면 이제 검증실패! 하고 띄워준다.


# 코드	핵심 역할
# accuracy_score()	전체 예측 중 정답 비율
# f1_score(average='macro')	클래스별 F1을 구해 동일한 가중치로 평균
# zip(test, pred)	문서 인덱스와 예측값을 짝지음
# if labels[i] != p	잘못 분류한 문서만 선택
# json.dumps()	Python 자료구조를 JSON 문자열로 변환
# write_text()	문자열을 파일에 저장
# np.ndenumerate()	배열의 인덱스와 값을 함께 순회
# plt.subplots()	그림과 그래프 영역 생성
# ax.imshow()	배열을 색상 이미지로 표시
# fig.savefig()	그래프 이미지 저장
# assert	TF-IDF 최대 오차 조건 확인