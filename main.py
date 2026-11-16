"""CLI entry point for TF-IDF review search and classification experiment."""
from __future__ import annotations

import argparse

from experiment import experiment
from search import search


def main() -> None:
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
