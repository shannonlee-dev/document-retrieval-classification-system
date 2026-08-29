# Stop-word Ablation

`DEFAULT_STOP_WORDS`의 효과를 같은 데이터와 분할에서 한 번 비교했다.

- Dataset: 20 Newsgroups의 `comp.graphics`, `rec.sport.baseball`, `sci.space`
- Split: stratified 80:20, `random_state=42` (train 2,290 / test query 573)
- Retrieval: held-out test document를 쿼리로, train document를 검색 대상으로 사용했다. 같은 카테고리 라벨을 relevant로 정의했다.

| Condition | Accuracy | Macro F1 | Precision@10 | MAP@10 | Vocabulary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default stop words | 0.9040 | 0.9031 | 0.8162 | 0.7569 | 21,662 |
| No stop words | 0.8935 | 0.8933 | 0.7759 | 0.7078 | 21,782 |
| Default minus none | +0.0105 | +0.0098 | +0.0403 | +0.0491 | -120 |

이 고정 seed 실험에서는 기본 불용어가 분류와 label-based retrieval 지표 모두에서 높았다. 다만 카테고리 라벨을 relevance의 대리 지표로 사용했으므로, 사람의 relevance judgment 기반 검색 품질을 뜻하지는 않는다. 다중 seed·유의성 검정도 수행하지 않았다.

실행: `uv run python scripts/run_stop_word_ablation.py`

원본 수치: `artifacts/reports/stop_word_ablation.json`
