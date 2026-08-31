# Stop-word Ablation

`DEFAULT_STOP_WORDS`의 비교 workflow는 현재 고정된 20-category loader를 소비한다. 분류는 stratified 80:20 분할(`random_state=42`)을 사용하고, retrieval은 held-out test 문서를 쿼리로 하며 train 문서를 검색 대상으로 한다. retrieval relevance는 같은 카테고리 라벨을 대리 지표로 정의한다.

이 workflow의 Python 구현은 각 쿼리마다 검색 말뭉치를 순회한다. 따라서 전체 category 범위에서 query-by-corpus retrieval 단계는 이차 시간 복잡도이며, 현 build에 포함하지 않는다.

이전 세 카테고리 비교 JSON `artifacts/reports/stop_word_ablation.json`은 현재 데이터 계약과 맞지 않아 제거했다. 전체 범위에서 workflow를 실행하거나 별도로 최적화하기 전에는 새로운 불용어 수치 비교를 주장하지 않는다.

실행: `uv run python scripts/run_stop_word_ablation.py`
