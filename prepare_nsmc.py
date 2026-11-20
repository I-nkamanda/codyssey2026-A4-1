"""Sample a balanced, cleaned subset of NSMC reviews into data/reviews.csv.""" #
import csv #csv 파일을 다루는 라이브러리다.
import random #난수 관련 라이브러리다. 수많은 NSMC 데이더 중에 일부를 랜덤으로 골라서 사용한다.
import re #정규식. 역시 구두점 
from pathlib import Path #바로 아래의 Path를 위해서 불러오는 라이브러리.

ROOT = Path(__file__).resolve().parent #현재 실행중인 파이썬 스크립트의 경로 = __file__ 이다. 
# 이걸 절대경로로 해석해서 그 파일이 들어있는 디렉토리를 ROOT로 지정. 파일을 읽고 쓸 아래 스크립트들의 기준점을 정해주는 셈이다.
SRC = ROOT / 'data/nsmc_ratings_test.txt' #읽어올 Source 파일의 위치. Root 이하 data라는 디렉토리를 까고 그 안에 nsmc_rating_test.txt라는 파일을 가리킴.
OUT = ROOT / 'data/reviews.csv' #결과 파일 저장 위치. 위와 동일

PII_PATTERN = re.compile(r'[\w.-]+@[\w.-]+|\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}') #개인정보(Personally Identifiable Information) 감지. 이메일 주소 형식 및 전화번호 형식을 감지해 낸다.  [뭐시기]@[어시기.저시기] 혹은 []-[]-[] 형식의 연속된 숫자들을 미리 설정해둠.
TOKEN_PATTERN = re.compile(r'[가-힣a-z0-9]+') #토큰을 뽑아내는, 단어 추출 규칙. 완성형 한글/영문 호문자/숫다가 1글자 이상 연속된 덩어리를 모두 토큰으로 추출함.


def is_valid_review(text: str) -> bool: #review가 valid한지 알아내는 함수
    if not text or not text.strip(): #text가 없거나 빈칸인 경우
        return False #False!
    if PII_PATTERN.search(text): # text 안에 개인정보 패턴이 보이는 경우
        return False #False!
    tokens = TOKEN_PATTERN.findall(text.lower()) #text를 소문자한 문자열 안에서 TOKEN_PATTERN에 매칭되는 모든 결과를 찾아 tokens라는 list로 반환한다.
    return len(tokens) >= 2 #근데 토큰 길이가 2개 이상인 리뷰만 남긴다. 
# 제외 예시: "굿"(토큰 1개-> 제외) "ㅋㅋㅋㅋㅋ"(토큰 0개 ->제외) "완전재미있어요"(토큰 1개-> 제외)
# 통과 예시: "진짜 굿"(["진짜"], ["굿"] -> 토큰 2개->통과)


def main(n_per_class: int = 1000, seed: int = 42) -> None: #클래스(긍정/부정) 당 뽑아올 리뷰 개수는 기본적으로 1000개, random을 실행할 때 seed 값은 기본 42로 둠 (관용적으로 자주 쓰는 고정 시드값 = 42)
    rng = random.Random(seed) #random 라이브러리에서 특정 시드(기본 42)를 사용하는 독립 난수 생서기 객체 'rng'를 만들어 줍니다.
    pos, neg = [], [] #pos(itive)와 neg(ative)한 리뷰를 빈 list로 정의한다.
    with SRC.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t') #csv다 보니까 tab을 기준 구분자(delimiter)로 각 행의 열 데이터를 나누겠다는 이야기. (\t = tab 역이스케이프 문자다.)
        for row in reader: #reader의 매행마다...
            doc = (row.get('document') or '').strip() # 현재 행의 document 열. 앞뒤의 공백을 뗀 리뷰 본문을 가져 온다. [or ''] 부분이 추가된 이유는 해당 열이 비어있거나 공백일때도 에러 없이 실행하기 위해서!
            label = row.get('label') #label(1, 0)으로 된 긍정/부정 정답
            if label not in {'0', '1'} or not is_valid_review(doc): #label이 0이나 1이 아닐 때, 그리고 추출된 토큰이 2개 미만일 때는
                continue #다음 줄로 스루스루 한다.
            item = (row['id'], doc, int(label)) #이제 row id, 본문, label(긍정/부정 평가)를 item이라는 tuple로 묶어준다. 앞으로 내용이나 구조를 바꿀 필요가 없는 data record기 때문에 튜플을 쓴다.
            (pos if label == '1' else neg).append(item) #label이 긍정(1)일 때는 pos에 append하고, 아닐 때는 neg에 append해준다.

    rng.shuffle(pos) #긍정 평가에 들어간 튜플들의 순서를 섞어준다. (배치 순서만)
    rng.shuffle(neg) #부정 평가에 들어간 튜플들의 순서도 섞어준다.
    sampled = pos[:n_per_class] + neg[:n_per_class] #이제 이렇게 섞인 positive 앞에서부터 n_per_class개, negative도 앞에서 n_per_class개씩 잘라내서 붙여 준다.
    rng.shuffle(sampled) #이제 그 결과물을 또 다시 섞어준다.

    with OUT.open('w', encoding='utf-8', newline='') as f: #OUT위치에다
        writer = csv.writer(f) #파일 객체 f에 CSV 형식으로 쓸 수 있는 writer 객체를 생성해둔 뒤,
        writer.writerow(['id', 'text', 'label']) #각 열의 header를 'id', 'text', 'label'로 지정해 두고
        writer.writerows(sampled) #sampled에 저장된, 섞인 tuple 값들을 쭈우욱 적어준다.

    print(f'Saved {len(sampled)} reviews ({n_per_class} pos / {n_per_class} neg) to {OUT}') #"xxxx 개의 리뷰 (nn 긍정 / nn 부정)을 OUT 위치에 저장했습니다" 하고 완료 메시지 출력을 해 준다.


if __name__ == '__main__': # 'python3 prepare_nsmc.py'와 같이 직접 호출되었을 시에는
    main() #main 함수를 실행한다.
