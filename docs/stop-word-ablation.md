# Stop-word Ablation

`DEFAULT_STOP_WORDS`의 비교 workflow는 고정된 20-category loader를 소비한다. 분류는 stratified 80:20 분할을 사용하고, retrieval은 held-out test 문서를 쿼리로 하며 train 문서를 검색 대상으로 한다. retrieval relevance는 같은 카테고리 라벨을 대리 지표로 정의한다.

## 10-seed 결과

`random_state=42..51` 10개 seed에서 매번 14,647개 train 문서로 vocabulary/IDF와 SVM을 학습하고 3,662개 held-out 문서를 평가했다. 표의 `±`는 seed 간 sample standard deviation이다.

| variant | Accuracy | macro F1 | Precision@10 | MAP@10 |
|---|---:|---:|---:|---:|
| default stop words | 0.769224 ± 0.005802 | 0.757830 ± 0.007531 | 0.518531 ± 0.003832 | 0.428169 ± 0.003947 |
| no stop words | 0.761251 ± 0.006674 | 0.749226 ± 0.009126 | 0.453332 ± 0.005391 | 0.368556 ± 0.005704 |
| default - none | +0.007974 ± 0.004155 | +0.008604 ± 0.005243 | +0.065199 ± 0.004799 | +0.059613 ± 0.004585 |

고정 불용어 적용은 네 지표 모두 10/10 seed에서 높았다. 평균 vocabulary는 불용어 적용 79,997개, 미적용 80,119개였다. seed별 원수치, 범위, 표준편차와 승패 수는 `artifacts/reports/stop_word_ablation.json`에 보존한다.

이 결과는 같은 카테고리 라벨을 relevance로 두는 현 대리 지표에 한정된다. 고정 불용어가 일반적 검색 의도와 모든 corpus에서 더 나은 것을 보장하지는 않는다.

## 실행 방법과 계산 한계

프로젝트의 기준 Python 구현은 각 쿼리마다 검색 말뭉치를 순회하므로 query-by-corpus retrieval이 이차 시간 복잡도를 갖는다. 10-seed 실험은 동일한 sparse dot-product 점수와 score/document-order stable tie-break를 유지한 배치 벡터화 평가로 가속했다. 벡터화 결과는 소규모 fixture에서 기준 구현과 일치함을 먼저 확인했다.

기준 single-seed workflow: `uv run python scripts/run_stop_word_ablation.py`. 이 명령은 `random_state=42` 한 번만 실행하며 10-seed 집계 JSON을 재현하지는 않는다.
