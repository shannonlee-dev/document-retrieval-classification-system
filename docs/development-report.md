# 고도화 개발 완료 보고서

## 1. 구현한 기능

- 공통 CSR Count Matrix와 IDF를 재사용하는 Raw / Log / Double TF-IDF. Raw 기본 동작 보존.
- 직접 NumPy Okapi BM25: positive IDF, TF 포화, 전체 토큰 길이, unique query terms, k1/b 검증.
- term posting 기반 candidate-only 검색과 bounded Top-K heap. Scan baseline 유지, 외부 ID 동점 정렬, 1e-12 일치 기준.
- 버전 3 공유 NPZ/JSON 인덱스, 설정 복원·체크섬·corpus/report fingerprint 및 index SHA 검사.
- 같은 split/query의 E1–E6: 분류·검색·latency·불용어·Top-5, per-class 지표/혼동 행렬/오분류 및 질의별 품질 차이.
- offline demo·compare·interactive·search, build/benchmark 분리, 패키지 모듈 실행과 기존 설치 CLI 호환.

## 2. 수정한 파일

**신규 코드/구성**

- `src/document_system/__main__.py`, `requirements.txt`, `MANIFEST.in`
- `src/document_system/{bm25,inverted_index,presentation}.py`
- `src/document_system/experiments/{tf_variants,retrieval_benchmark}.py`
- `tests/test_{tf_variants,bm25,inverted_index,retrieval_artifacts,retrieval_benchmark,demo_cli}.py`
- `docs/baseline-analysis.md`, `docs/demo-output.txt`, 이 보고서, 설계/실행 계획과 신규 CSV/JSON/PNG 보고서

**기존 수정**

- `tfidf.py`, `validation.py`, `search.py`, `artifacts.py`, `cli.py`, `pipeline.py`, `build_stages.py`
- `dataset.py`: loader를 바꾸지 않고 fingerprint helper 추가
- `experiments/{metrics,ablation,reporting}.py`
- `README.md`, `Makefile`, `scripts/check.py`, `tests/test_artifacts.py`, `.github/workflows/validate.yml`

`sparse_matrix.py`, 분류 모델/평가/시각화, privacy 파이프라인, DATASET_LICENSE.md, pyproject.toml, uv.lock은 보존했다. CI는 기존 검증 명령을 유지하고 Python 3.10/3.13 matrix로 확장했다.

## 3. 실행 방법

```bash
pip install -r requirements.txt
python -m document_system build
python -m document_system benchmark
python -m document_system demo
python -m document_system interactive
python -m document_system compare --query "space shuttle orbit"
python -m document_system --query "space shuttle orbit" --topk 5
python -m document_system search --query "space shuttle orbit" --tf-mode log --strategy inverted
python -m document_system search --query "space shuttle orbit" --method bm25
uv run --frozen document-system --query "space shuttle orbit" --topk 5
make demo
make benchmark
```

현재 workspace의 전체 corpus 인덱스와 benchmark가 준비되어 있다. Runtime은 Git 제외 경로이므로 새 clone에서는 build/benchmark를 준비한다. Demo는 네트워크·학습을 시작하지 않는다.

## 4. 테스트 결과

| 검증 | 결과 |
|---|---|
| 변경 전 pytest | 83 passed |
| make test (Python 3.13.7) | 154 passed |
| pytest (Python 3.10.22 / 별도 잠금 환경) | 154 passed |
| make smoke | 3 passed |
| make check | 문법·문서/Ruff/format 통과 |
| make build | wheel + sdist 생성 성공 |
| 새 프로세스 demo / 기존 CLI / BM25 search | 정상 종료 |
| 독립 코드 리뷰 | 주요 3개 문제 수정, 리뷰 재검증 28 passed |

수치 비교 Raw/Log는 sklearn, Double/BM25는 수작업 reference를 사용했다. Save/load, 오류·OOV·Top-K·tie, train/test 분리, seed 재현성, CSV/JSON schema와 미측정 latency null도 검증했다. CI는 dataset을 다운로드하지 않는다.

이 환경에서는 make를 /tmp에 내려받아 사용했다. 네트워크 제한으로 setuptools resolution이 처음 실패했으나 의존성을 준비한 뒤 offline cache로 최종 패키지 빌드가 성공했다. GitHub 원격 CI 자체는 실행하지 않았으며 로컬에서 동일 명령과 두 Python 버전을 검증했다.

## 5. 벤치마크 결과

전체 corpus 18,309문서·20개 주제, stratified 80/20(14,647/3,662), seed 42. 분류는 test 전체, 검색 품질은 고정 held-out subset 100개. 1개 빈/OOV 질의를 품질에서 제외하지 않았고, 반복 latency는 공통 유효 질의 99개 × 5 = 495표본이다.

| TF | Accuracy | Macro F1 | P@10 | MAP@10 |
|---|---:|---:|---:|---:|
| raw | 0.767067 | 0.754385 | 0.521000 | 0.434458 |
| log | 0.770071 | 0.757423 | 0.557000 | 0.478212 |
| double | 0.768979 | 0.753667 | 0.528000 | 0.456887 |

| 검색 방식 | P@10 | MAP@10 | Mean ms | P95 ms |
|---|---:|---:|---:|---:|
| tfidf | 0.521000 | 0.434458 | 12.546 | 31.254 |
| bm25 | 0.570000 | 0.490718 | 12.546 | 31.088 |

| 문서 수 | Scan mean ms | Inverted mean ms | Scan P95 ms | Inverted P95 ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1000 | 9.896 | 0.308 | 10.206 | 0.415 | 32.09x |
| 5000 | 48.501 | 0.488 | 49.639 | 0.604 | 99.44x |
| 10000 | 97.403 | 0.712 | 99.111 | 0.970 | 136.90x |
| 18309 | 175.790 | 1.133 | 178.284 | 1.592 | 155.10x |

속도 실험은 품질 실험과 별도로 짧은 질의 3개 × 5회, warm-up 뒤 perf_counter 측정이다. 모든 규모에서 동일 결과를 확인했다. 구축 비용·추가 메모리는 CSV/JSON에 기록했고 로드 시간은 검색 시간에 넣지 않았다.

AP@10 기준 BM25 우세 55 / TF-IDF 우세 32 / 동률 13. ID 1264는 Raw AP 0.3267 / BM25 AP 1.0, ID 18749는 Raw 0.8354 / BM25 0.3796이다. 따라서 이번 설정의 평균 개선을 모든 질의의 우수성으로 일반화하지 않는다.

불용어 On/Off: Accuracy 0.767067 / 0.762425, Macro F1 0.754385 / 0.752118, P@10 0.521 / 0.465, MAP@10 0.434458 / 0.386463. 기존 10-seed 결과는 별도로 보존했다.

세 대표 질의의 Top-5 ID·순위·카테고리·token 길이, 새 오분류 5건 이상, per-class 지표와 혼동 행렬은 [실험 요약](../artifacts/reports/experiment_summary.md)과 [전체 JSON](../artifacts/reports/retrieval_benchmark.json)에 있다.

질의별 상세 ranking은 전체 실험 뒤 동일한 split/query ID에서 독립적으로 재계산하고 aggregate와 1e-12 이내 일치를 확인했다. 리뷰 수정된 E5의 외부 ID 동점 규칙으로 품질도 재계산했으며 이 corpus의 기존 E5 수치와 같았다. 이 과정에서 이미 측정된 분류·latency 수치는 그대로 보존했고 provenance를 JSON에 기록했다.

## 6. CLI Demo 결과

순서: 개요 → [1/6] Raw 검색 → [2/6] TF 세 방식의 Top-5/품질 → [3/6] BM25 순위/길이/snippet 비교 → [4/6] Scan/Inverted 일치·반복 통계 → [5/6] 분류/오분류/혼동 행렬 경로 → [6/6] 수치·일치 검증 요약.

실제 `demo` 명령 출력 일부:

```text
Query      : space shuttle orbit
Method     : TF-IDF
TF Mode    : Raw
Index      : Full Scan
Top-K      : 5
1    | 0.413791 | 4672        | sci.space
2    | 0.384463 | 4389        | sci.space

Full Scan      | 175.790 | 175.736 | 178.284
Inverted Index | 1.133   | 0.979   | 1.592
Speedup: 155.10x
Result Consistency: PASS
Demo Completed.
```

위 speed 표는 Mean/P50/P95 ms이며 해당 실행의 현재 single-query 시간과 구분하여 출력한다. 전체 원본 출력은 [demo-output.txt](demo-output.txt)에서 확인한다.

## 7. 남은 제한사항

- 구현·필수 실험에 미완료 항목은 없다. 대규모 품질 평가는 test 전체 3,662개가 아닌 명시된 subset 100개다. 전체 test 질의 평가는 이번 실행에서 측정하지 않았다.
- 신규 E5는 seed 42 한 번이다. 새 다중-seed E1–E4, BM25 파라미터 sweep, 통계적 신뢰구간·실제 semantic qrels는 이번에 측정하지 않았다.
- 속도는 이 환경/질의에서의 값이며, posting 메모리는 NumPy payload만 포함한다. 긴/빈번한 단어 질의는 많은 후보를 건드린다.
- 카테고리 relevance는 의미적 관련성의 대리값이다. BoW의 어순/문맥 한계와 주제 공통 어휘에 의한 오분류가 남는다.
- 기존 structured redaction을 유지했다. 자유 형식 개인정보가 전부 제거됐다는 주장을 하지 않는다.
- Runtime/cache는 Git에 포함하지 않는다. 현재 브랜치에 작업 내용을 유지했으며 push/PR/원격 CI 실행은 하지 않았다. 후속 요청에 따라 변경사항을 기능별 커밋으로 정리했다.
