# Stop-word Ablation

`DEFAULT_STOP_WORDS`의 효과를 10개 seed에서 비교했다.

- Dataset: 20 Newsgroups의 `comp.graphics`, `rec.sport.baseball`, `sci.space`
- Seeds: `0`–`9`; 각 seed에서 stratified 80:20 split (train 2,290 / test query 573)
- Retrieval: held-out test document를 쿼리로, train document를 검색 대상으로 사용했다. 같은 카테고리 라벨을 relevant로 정의했다.

| Condition | Accuracy (mean ± sd) | Macro F1 (mean ± sd) | Precision@10 (mean ± sd) | MAP@10 (mean ± sd) | Vocabulary (mean ± sd) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default stop words | 0.9187 ± 0.0080 | 0.9187 ± 0.0079 | 0.8039 ± 0.0050 | 0.7412 ± 0.0061 | 21,707.7 ± 361.6 |
| No stop words | 0.9045 ± 0.0163 | 0.9045 ± 0.0164 | 0.7765 ± 0.0053 | 0.7073 ± 0.0076 | 21,828.4 ± 361.4 |
| Default minus none (paired) | +0.0141 ± 0.0169 | +0.0142 ± 0.0171 | +0.0273 ± 0.0041 | +0.0339 ± 0.0063 | -120.7 ± 0.5 |

10개 paired seed의 평균에서 기본 불용어가 분류와 label-based retrieval 지표 모두에서 높았다. 다만 카테고리 라벨을 relevance의 대리 지표로 사용했으므로, 사람의 relevance judgment 기반 검색 품질을 뜻하지는 않는다. 유의성 검정과 사람의 relevance judgment 평가는 수행하지 않았다.

실행: `uv run python scripts/run_stop_word_ablation.py`

원본 수치: `artifacts/reports/stop_word_ablation.json`
