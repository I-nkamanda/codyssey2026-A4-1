"""Build a reproducible, synthetic and anonymous set of short movie reviews."""
import csv
import random
from pathlib import Path

rng = random.Random(42) #역시 관행적으로 seed는 42로 설정해 놓았다. (역사와 전통))
subjects = ['연기', '스토리', '음악', '영상', '대사', '연출', '결말', '액션', '전개', '인물'] # 리뷰의 주요 소재 pool!
positive = ['좋았다', '인상적이었다', '만족스러웠다', '훌륭했다', '마음에 들었다', '재미있었다'] # 긍정 서술어 pool!
negative = ['아쉬웠다', '지루했다', '실망스러웠다', '별로였다', '마음에 들지 않았다', '어색했다'] # 부정 단어 pool!
contexts = ['극장에서 봤는데', '기대하고 봤는데', '주말에 봤는데', '끝까지 봤는데', '다시 봤는데', '처음 봤는데'] #컨텍스트 pool!
connectors = ['그리고', '반면', '하지만', '또한'] #접속사 pool!
rows = [] # 빈 list row를 만들어 준다.
for i in range(1200): #1200번 반복 생성합니다.
    label = i % 2 # 한 번은 긍정, 한 번은 부정으로 (각 600개씩 생성된다)
    main = positive if label else negative #정답 예시를 만들어주고
    other = negative if label else positive #오답도 일부러 만들어줌(label과 반대)
    # Mixed sentiment and occasional inversion make the classification problem nontrivial.
    a, b = rng.sample(subjects, 2) #subjects 리스트에서 서로 다른 2개를 골라서 a, b로 명명
    first, second = rng.choice(main), rng.choice(other if rng.random() < .70  else main) #first는 정답 목록에서 고르고, second는 rng.random()으로 0~1 사이의 실수를 생성한 뒤 그게 0.7 미만이면(70% 확률) 정답과 반대되는 표현(other)을, 나머지 경우(30% 확률) 정답과 같은 표현(main)이 설정됨
    if rng.random() < .45: #0이상 1미만의 무작위 실수를 생성하는데 그게 0.45 미만이면(45% 확률로)
        text = f'{rng.choice(contexts)} {a} {first} {rng.choice(connectors)} {b} {second}' #main에서 선택한 first 를 앞에 배치한다.
        #예시: 생성된 리뷰 = 극장에서 봤는데 연기 좋았다 하지만 결말 아쉬웠다
    else: #아니라면 (55% 확률로)
        text = f'{rng.choice(contexts)} {b} {second} {rng.choice(connectors)} {a} {first}' #second를 앞에, first를 뒤에 배치한다.
        #예시: 생성된 리뷰 = 극장에서 봤는데 결말 아쉬웠다 하지만 연기 좋았다
    rows.append((f'review-{i+1:04d}', text, label)) #rows에 붙여준다.
out = Path(__file__).resolve().parent / 'data/reviews_synthetic.csv' #path 하위 위치 /data 에 reviews_synthetic.csv 라는 이름으로 출력 위치 설정
out.parent.mkdir(exist_ok=True) #없으면 path/data 디렉토리를 만들어 주고, 이미 해당 폴더가 있더라도 중복 오류를 만들지 않는다.
with out.open('w', encoding='utf-8', newline='') as f: #이제 원하는 위치에 쓰기모드(없으면 쓰고 있으면 덮어씌움) .csv 파일을 작성하는데
    w = csv.writer(f); w.writerow(['id', 'text', 'label']); w.writerows(rows) #각 열의 이름은 id, text, label 순서로 지어준다. 그 뒤에 rows 내용을 쭈우욱 써준다.

# 생각해보기: 합성 데이터의 한계를 지어봅시다. - 우선, 정답을 먼저 정해놓고 문장을 만들게 됩니다.
# 예컨대 데이터 처리 테스트 및 예측 파이프라인 작동을 확인시키는 데는 쓸만하지만, 실제 리뷰 감성 분류의 성능, 한국어 리뷰를 자연스럽게 생성한다든가 하는 일에는 큰 한계가 있는 것입니다.
# id의 홀/짝에 따라서 label 예측이 너무 쉽지요? 그러기 때문에 id까지 학습에 포함한다면 누출이 큽니다.