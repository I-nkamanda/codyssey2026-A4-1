"""CLI entry point for TF-IDF review search and classification experiment."""
from __future__ import annotations

import argparse #뒤의 argument를 쉽게 쓰게 해 주는 argparse 라이브러리를 가져옴

from experiment import experiment #experiment.py에서 experiment 함수를 불러옴
from search import search #search.py 에서 search 함수를 불러옴

# parser.add_argument() 는 어떤 명령행 option을 받을 지 정의해준다.
def main() -> None:
    parser = argparse.ArgumentParser(description='TF-IDF review search')
    parser.add_argument('--query', default='연기 감동') #python3 main.py --query '연기 감동' 
    parser.add_argument('--topk', type=int, default=5) #python3 main.py --topk 3 와 같이 씀. 정수 인수를 받습니다.
    parser.add_argument('--experiment', action='store_true') #python3 main.py --experiment 
    args = parser.parse_args() #실제로 전달된 옵션들을 해석한 뒤에 args 객체에 저장한다.
    if args.topk < 1: #topk 점수가 1 미만이라면
        parser.error('--topk must be positive') #상위 0, -1, -2 등은 말이 안 되므로 error를 띄워줍니다.
    if args.experiment: # experiment argument가 들어온다면
        experiment() #experiment를 실행해 줍니다.
    else: #이외에는 유사도 검색일 것이기 때문에
        results = search(args.query, args.topk) #query문의 topk 순위를 뽑아본 다음에
        if not results: #결과가 안 나오면
            print('일치하는 문서가 없습니다.') #없습니다! 하고 메시지를 내고
        for score, doc_id, snippet in results: #결과 객체 안의 점수/리뷰ID/원문 snippet(최대 120자)를 
            print(f'[{score:.4f}] {doc_id} {snippet}') #점수만 소숫점 4자리까지로 반올림해 표시 후에 쭉쭉 반환함.

# action = 'store_true' 가 있으면: --experiment=True 와 같이 옵션을 입력할 필요 없이 --experiment 문자열이 있으면 그대로 True로 받아들임
if __name__ == '__main__':
    main()
