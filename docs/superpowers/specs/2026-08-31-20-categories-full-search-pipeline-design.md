# 20-Category Full-Corpus Search Pipeline Redesign

## Goal

20 Newsgroups의 세 카테고리 제한을 제거하고 전체 20개 카테고리를 고정 사용한다. 분류 평가는 데이터 누수가 없는 80/20 TF-IDF feature space를 유지하며, 검색은 전체 문서 100%에 별도로 fit한 TF-IDF feature space와 행렬을 사용한다.

## Scope

이 변경은 다음을 포함한다.

1. 실제 데이터 loader와 build 진입점에서 20개 카테고리 계약을 강제한다.
2. 분류용 TF-IDF와 검색용 TF-IDF를 독립된 build stage로 분리한다.
3. 검색 runtime artifact를 전체 corpus fit 형식으로 전환하고 버전을 올린다.
4. 보고서와 CLI에서 두 vocabulary 및 fit 범위를 구분한다.
5. 관련 테스트, README와 생성 산출물을 새 파이프라인 의미에 맞춘다.

검색 알고리즘, 전처리 규칙, SVM 설정, privacy sanitization 규칙과 CLI 명령 형식은 변경하지 않는다. Stop-word ablation의 검색 평가 성능 최적화도 이번 범위에 포함하지 않는다.

## Fixed Decisions

- 데이터셋은 선택형 설정 없이 20 Newsgroups의 전체 20개 카테고리를 고정 사용한다.
- 분류 TF-IDF는 80% 학습 분할에만 fit하고 20% 테스트 분할은 transform만 한다.
- 검색 TF-IDF는 정제 후 유지된 전체 문서 100%에 별도로 fit한다.
- 분류 vectorizer와 검색 vectorizer는 인스턴스와 vocabulary/IDF를 공유하지 않는다.
- runtime artifact에는 검색 vectorizer와 검색 행렬만 저장한다.
- `sci.med`를 포함함에 따라 자유형 건강정보가 ML 입력 또는 제한 snippet에 남을 수 있는 잔여 위험을 수용한다. 현재 structured-PII redaction과 240자 snippet 정책은 그대로 유지한다.

## Dataset Contract

`dataset.py`는 `fetch_20newsgroups` 호출에서 `categories` 인자를 제거한다. `subset="all"`, metadata 제거, shuffle과 random state는 유지한다.

실제 20 Newsgroups build는 다음 조건을 검증한다.

- `target_names`가 정확히 20개다.
- sanitization 후 labels에 20개 class ID가 모두 남아 있다.
- class ID가 `target_names`의 `0..19` 범위와 일치한다.
- privacy report에 20개 카테고리의 처리 통계가 모두 존재한다.

`build_project()`는 loader가 테스트나 외부 호출에서 대체되더라도 이 계약을 다시 확인한다. 반면 `build_from_dataset()`은 소규모 fixture와 재사용 가능한 pipeline seam을 위해 두 개 이상의 카테고리를 가진 사용자 정의 `DatasetBundle`을 계속 허용한다.

## Pipeline Architecture

`pipeline.py`는 데이터 로드, 두 stage 호출, 보고서 기록과 최종 `BuildReport` 조립만 담당한다. 새 `build_stages.py`는 분류·검색 stage 함수와 각 result dataclass를 소유한다. 기존 `classification.py`, `search.py`, `tfidf.py`는 알고리즘 계층으로 유지하며 build orchestration을 알지 못한다.

```text
sanitized 20-category DatasetBundle
                 |
                 +--> classification stage
                 |      stratified 80/20 split
                 |      classification vectorizer.fit(80%)
                 |      train/test matrices
                 |      sklearn equivalence validation on 80%
                 |      SVM training and held-out evaluation
                 |
                 +--> search stage
                        search vectorizer.fit_transform(100%)
                        sklearn equivalence validation on 100%
                        search examples
                        versioned runtime artifact
```

두 stage는 작은 result dataclass를 통해 필요한 결과만 반환한다. 분류 stage result에는 평가 결과, 검증값, stage example, 분할 수와 vocabulary 크기만 남기고 train/test 행렬과 vectorizer는 반환하지 않는다. 따라서 검색용 전체 행렬을 만들기 전에 분류 대형 행렬이 해제될 수 있다.

검색 stage result는 artifact 저장에 필요한 검색 vectorizer와 전체 검색 행렬, 검증값, 검색 예시와 통계를 보관한다. 검색은 count matrix가 필요한 stage example을 만들지 않으므로 `fit_transform()`을 사용해 전체 corpus count matrix의 수명을 최소화한다.

## Classification Stage

분류 stage는 기존 평가 방법을 유지한다.

1. 전체 row ID를 고정 random state와 stratification으로 80/20 분할한다.
2. 분류 vectorizer를 학습 문서 80%에만 fit한다.
3. 테스트 문서 20%는 같은 vectorizer로 transform한다.
4. 학습 행렬을 scikit-learn 기준 구현과 비교해 최대 오차 `1e-6` 이하를 확인한다.
5. 학습 행렬로 linear SVM을 학습하고 held-out 행렬에서 Accuracy, macro F1, confusion matrix와 오분류 예시를 계산한다.

테스트 분할에만 있는 token은 분류 vocabulary에 들어가지 않아야 한다. 20개 카테고리의 불균형 영향을 확인할 수 있도록 Accuracy와 함께 macro F1을 계속 핵심 지표로 사용한다.

## Search Stage

검색 stage는 분류 분할을 입력으로 사용하지 않는다.

1. 독립된 검색 vectorizer를 생성한다.
2. 정제 후 유지된 전체 문서 100%에 `fit_transform()`한다.
3. 검색 행렬을 전체 corpus 기준 scikit-learn 결과와 비교한다.
4. 동일한 검색 vectorizer로 기본 query를 transform하고 Top-k 예시를 만든다.
5. 검색 vectorizer, 전체 행렬, labels, 20개 target names, source document IDs와 제한 snippet을 runtime artifact로 저장한다.

검색 matrix의 행 순서는 `DatasetBundle`의 texts, labels, source document IDs와 동일하게 유지한다. 검색 query와 corpus는 반드시 검색 전용 vocabulary/IDF를 공유한다.

## Runtime Artifact Compatibility

`ARTIFACT_VERSION`을 `2`로 올린다. 기존 artifact는 행 수가 전체 문서와 같더라도 vectorizer가 80%에 fit되었을 수 있으므로 호환 대상으로 취급하지 않는다.

`search_index_data.json`에 다음 의미 필드를 추가한다.

- `fit_scope`: 고정값 `full_corpus`
- `fit_document_count`: 검색 vectorizer fit에 사용한 전체 문서 수
- `category_count`: 고정 데이터 build에서는 `20`

loader는 다음을 검증한다.

- artifact version이 `2`다.
- `fit_scope`가 `full_corpus`다.
- `fit_document_count`가 matrix 행 수와 같다.
- `category_count`가 `target_names` 수와 같다.
- labels가 target name 범위 안에 있다.
- matrix, snippets, labels와 document IDs의 행 수가 일치한다.

기존 v1 artifact에는 재빌드 안내 오류를 반환한다. `python main.py --query ...` 사용법은 유지한다.

## Reports and CLI

기존 report 경로는 유지하되 내부 의미를 명시한다.

- `classification_metrics.json`: 분류 결과, 80/20 분할 정보와 `classification_vocabulary_size`
- `search_index_statistics.json`: 검색 행렬 통계, `fit_scope`, `fit_document_count`와 `search_vocabulary_size`
- `tfidf_sklearn_validation.json`: `classification`과 `search` 검증 결과 및 각 fit 범위
- `tfidf_transformation_example.json`: 분류 학습 stage의 예시임을 나타내는 `fit_scope`
- `search_result_examples.json`: 100% 검색 feature space에서 생성된 Top-k 예시
- `dataset_sanitization_report.json`, `classification_error_examples.json`, `classification_confusion_matrix.png`: 20개 카테고리 데이터 기준 결과

`BuildReport`의 모호한 필드는 다음처럼 분리한다.

- `vocabulary_size` -> `classification_vocabulary_size`, `search_vocabulary_size`
- `validation_passed` -> `classification_validation_passed`, `search_validation_passed`
- `max_absolute_error` -> `classification_max_absolute_error`, `search_max_absolute_error`

CLI build 완료 메시지도 두 vocabulary와 두 최대 오차를 각각 출력한다. build와 search 명령의 인자 형식은 유지한다.

## Privacy Boundary

카테고리 확대 외의 privacy 동작은 변경하지 않는다.

- headers, footers와 quotes 제거
- email, phone, URL, IPv4와 IPv6 structured redaction
- sanitization 후 빈 문서 제외
- 정제된 전체 문서 비저장
- artifact와 report에 최대 240자의 재검증된 snippet만 저장

전체 20개에는 `sci.med`가 포함된다. 현재 sanitizer는 자유형 사람 이름, 주소 또는 건강정보를 탐지하거나 제거하지 않으므로 PII-free를 보장하지 않는다. 사용자는 이 잔여 위험을 명시적으로 수용했으며, 이번 변경에서는 category별 snippet 억제나 추가 detector를 도입하지 않는다.

## Stop-Word Ablation

Ablation은 모델 비교의 공정성을 위해 기존 held-out 구조를 유지한다. vectorizer는 각 variant별 학습 80%에만 fit하며 테스트 20%를 query로 사용한다. 데이터셋 설명은 세 카테고리 문자열에서 전체 20 Newsgroups로 수정한다.

현재 retrieval 평가는 query와 corpus의 Python 이중 루프이므로 20개 전체 데이터에서는 실행 시간이 크게 증가할 수 있다. 계산 방식 최적화나 sampling은 결과 정의를 바꿀 수 있어 별도 작업으로 남긴다.

## Error Handling

- 실제 build에서 카테고리가 정확히 20개가 아니거나 한 카테고리가 sanitization 후 사라지면 중단한다.
- 분류 또는 검색 TF-IDF의 scikit-learn 대조가 허용 오차를 넘으면 artifact를 저장하지 않는다.
- 검색 matrix와 metadata 수가 불일치하면 저장 또는 로드를 거부한다.
- v1 또는 필수 fit-scope metadata가 없는 artifact는 재빌드를 안내한다.
- blank query와 검색 vocabulary에 없는 query의 기존 오류 동작은 유지한다.

## Testing

변경은 테스트 우선으로 진행한다.

1. loader가 category filter 없이 `subset="all"`을 요청하는 테스트
2. 실제 build 계약이 20개 미만 target names, 누락된 class ID와 범위 밖 label을 거부하는 테스트
3. privacy report가 20개 카테고리 통계를 포함하는 loader 테스트
4. 고정된 split에서 테스트 전용 token이 분류 vocabulary에는 없고 검색 vocabulary에는 있는 pipeline 테스트
5. 분류 검증 행 수가 train count와 같고 검색 검증 행 수가 전체 document count와 같은 테스트
6. 검색 matrix, snippets, labels와 document IDs가 전체 문서 수와 일치하는 테스트
7. artifact v2 round-trip, v1 거부, fit scope/count/category/label 검증 테스트
8. 20x20 confusion matrix 생성과 report 의미 필드 테스트
9. CLI build 출력이 분류와 검색 vocabulary를 구분하고 기존 search 명령이 v2 artifact를 읽는 테스트
10. 전체 unit test, lint와 실제 20-category build 검증

실제 build 검증에서는 문서 수, 20개 카테고리, 80/20 수, 두 vocabulary 크기, 두 TF-IDF 오차, 검색 matrix shape, Accuracy, macro F1 및 runtime artifact 재로딩을 확인한다.

## Documentation and Generated Outputs

README의 구조도, 3-category 설명, 데이터 수, vocabulary, matrix shape, 분류 지표, confusion matrix 설명과 과거 기준선 구간을 현재 20-category dual-feature-space 파이프라인에 맞춘다. `DATASET_LICENSE.md`의 데이터 범위와 privacy 설명도 전체 20개 및 명시적으로 수용된 잔여 위험과 일치시킨다.

실제 build가 성공한 뒤 `artifacts/runtime/*`와 `artifacts/reports/*`를 새 결과로 갱신한다. 이전 3-category 보고서가 새 설계 설명과 함께 남지 않도록 추적된 산출물 전체의 일관성을 확인한다.

## Acceptance Criteria

- 실제 loader와 build는 전체 20개 카테고리를 고정 사용한다.
- 분류 vectorizer는 train 80%에만 fit되고 held-out 20%는 transform만 된다.
- 검색 vectorizer와 matrix는 정제 후 전체 문서 100%에 fit된다.
- 두 vectorizer는 독립된 vocabulary와 IDF를 가진다.
- 검색 전용 token 포함 여부를 통해 100% fit 동작이 회귀 테스트로 증명된다.
- 분류와 검색 TF-IDF 모두 scikit-learn 대비 최대 오차 `1e-6` 이하이다.
- runtime artifact v2가 full-corpus fit 범위를 검증하고 v1을 거부한다.
- 보고서와 CLI가 두 feature space를 혼동 없이 표시한다.
- 현재 privacy 처리 방식과 CLI 사용법은 유지된다.
- 관련 문서와 생성 산출물이 20-category 결과와 일치한다.
- 전체 테스트와 실제 build 검증이 통과한다.
