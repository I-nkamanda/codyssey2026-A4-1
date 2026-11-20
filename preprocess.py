"""Text preprocessing, tokenization, and CSV dataset loading."""
from __future__ import annotations # type 어노테이션에 사용된 표현식의 평가를 미뤄줌.
# 타입 어노테이션(Type Hint)을 즉시 평가하지 않고
# 나중에 평가할 수 있도록 처리 방식을 변경한다.
# 아직 정의되지 않은 클래스를 타입 힌트로 참조할 때 유용하다.

import csv #csv 라이브러리를 불러오고, 
import re # 정규식 라이브러리도 불러옵니다.
from pathlib import Path #pathlib 라이브러리에서 Path도 불러옵니다.

ROOT = Path(__file__).resolve().parent #현재 실행되는 스크립트의 위치를 기준으로 ROOT를 해 주고
STOPWORDS = {'이', '그', '저', '것', '수', '정말', '너무', '그냥', '그리고', '하지만', '영화'} #불용어들을 집합으로 정의합니다. 왜 집합을 쓰냐면 어떤 단어가 포함되어 있는지 빠르게 검사하기 위해서입니다.
TOKEN = re.compile(r'[가-힣a-z0-9]+') #토큰이란 한글/알파벳/숫자 등으로 구성된 것이라고 정의합니다.


def tokenize(text: str) -> list[str]: #토크나이제이션 함수는 문장(string)받아서 list로 만들어줍니다.
    """Lowercase text, extract alphanumeric/Hangul tokens, and remove stopwords."""
    return [w for w in TOKEN.findall(text.lower()) if w not in STOPWORDS] #입력 문자열 text를 소문자로 변환하고, 정규표현식으로 토큰 패턴을 모두 찾아서 토큰을 추출해줍니다.
    # 각 토큰 w 가 불용어 집합에 없으면 결과 list에 포함해서 반환합니다.


def load_data(path: Path | None = None) -> list[dict[str, str]]: #.csv에 저장되어 있는 데이터를 로딩한 뒤 각 행을 딕셔너리로 구성한 리스트를 반환하빈다.
    """Load review rows from a UTF-8 CSV file with id, text, and label columns."""
    target = path or (ROOT / 'data/reviews.csv') #따로 설정한 path가 없으면 data/reviews.csv를 기본적으로 읽어옵니다.
    with target.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f)) #csv.DictReader로 읽어온 것을 list화해서 반환합니다.
