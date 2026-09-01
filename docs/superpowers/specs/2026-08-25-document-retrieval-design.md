# 문서 검색·분류 시스템 설계

## 1. 목표와 범위

20 Newsgroups의 20개 카테고리와 18,846개 문서를 모두 사용해 다음을 재현 가능한 CLI 프로젝트로 구현한다.

- 영문 텍스트 정제, 토큰화, 불용어 제거
- NumPy만으로 TF, IDF, TF-IDF 핵심 계산 구현
- Scikit-learn과 최대 절대 오차 `1e-6` 이내 검증
- NumPy 코사인 유사도 기반 Top-5 문서 검색
- TF-IDF 기반 선형 SVM 분류와 성능 평가
- 희소 표현의 수학적 동등성과 메모리 이점 분석
- 혼동 행렬, 오분류 사례, 검색 예시 및 3단 구성 README

문서나 어휘를 임의로 샘플링하지 않는다. 학습·평가 분할은 전체 데이터를 계층화 8:2로 나누고 `random_state=42`를 고정한다.

## 2. 핵심 설계 결정

### 2.1 NumPy 희소 행렬

TF-IDF 행렬은 CSR과 같은 세 배열로 직접 표현한다.

- `data`: 0이 아닌 값, `float64`
- `indices`: 각 값의 vocabulary 열 번호, `int32`
- `indptr`: 문서별 시작 위치, `int32` 또는 실제 `nnz`가 범위를 넘으면 `int64`
- `shape`: `(문서 수, vocabulary 크기)`

문서 `d`에 단어 `t`가 없으면 `TF(t,d)=0`이고 `TF-IDF(t,d)=0`이다. L2 정규화도 0을 바꾸지 않으므로, 0인 원소를 저장하지 않는 희소 표현은 밀집 행렬과 수학적으로 동일하며 샘플링이나 근사가 아니다.

실행 시 다음 값을 측정한다.

```text
density = nnz / (document_count * vocabulary_size)
sparsity = 1 - density
dense_bytes = document_count * vocabulary_size * 8
sparse_bytes = data.nbytes + indices.nbytes + indptr.nbytes
compression_ratio = dense_bytes / sparse_bytes
```

이 결과는 JSON 로그와 README에 기록한다. 일부 행에 대해서는 희소 행렬을 밀집 배열로 복원해 직접 계산 결과가 동일한지도 테스트한다.

### 2.2 TF-IDF 정의

학습 문서 수를 `N`, 단어 `t`가 등장한 학습 문서 수를 `df(t)`라고 한다.

```text
TF(t, d) = count(t, d)
IDF(t) = log((1 + N) / (1 + df(t))) + 1
TF-IDF(t, d) = TF(t, d) * IDF(t)
normalized(d) = TF-IDF(d) / ||TF-IDF(d)||_2
```

Scikit-learn 검증 설정은 다음과 같이 고정한다.

```text
smooth_idf=True
sublinear_tf=False
norm="l2"
use_idf=True
dtype=float64
lowercase=False
fixed vocabulary와 동일한 analyzer 사용
```

전처리된 토큰과 vocabulary를 양쪽 구현에 동일하게 전달한다. Scikit-learn의 희소 결과를 밀집 변환하지 않고 행별 열 인덱스와 값을 정렬해 비교하며, 전체 원소의 최대·평균 절대 오차와 PASS/FAIL을 남긴다.

### 2.3 분류 모델

Scikit-learn의 `SGDClassifier(loss="hinge")`를 사용한다. 이는 결정 함수 `w·x+b`와 hinge loss를 사용하는 선형 SVM이다. 사용자 정의 `data`, `indices`, `indptr` NumPy 배열은 TF-IDF를 다시 계산하지 않고 Scikit-learn이 받는 SciPy CSR 컨테이너의 버퍼로 연결한다. SciPy는 이 분류 입력 경계에서만 사용하며 TF, IDF, 정규화와 검색 계산은 계속 NumPy 직접 구현을 사용한다.

- 분할: stratified 8:2
- 난수: `random_state=42`
- 학습 데이터에만 vocabulary와 IDF 적합
- 테스트 데이터에는 학습 vocabulary와 IDF로 `transform`만 수행
- 배치 순서와 epoch 수를 고정해 재현성 확보
- Accuracy, macro F1, confusion matrix 산출

학습과 예측은 희소 행을 배치로 전달하므로 전체 또는 배치 밀집 행렬을 생성하지 않는다. 평가 로그에는 모델 클래스뿐 아니라 선형 SVM으로 보는 수학적 근거, CSR 입력 경계와 실제 파라미터를 명시한다.

## 3. 구성 요소

```text
main.py
src/document_system/
  preprocessing.py   정제, 토큰화, 불용어 제거
  sparse_matrix.py   NumPy 희소 행렬 자료구조와 행/배치 변환
  tfidf.py           vocabulary, TF, IDF, 정규화, transform
  validation.py      sklearn 교차 검증과 단계별 출력
  search.py          쿼리 변환, 희소 코사인 유사도, Top-k
  classification.py  분할, 배치 선형 SVM 학습, 지표와 오분류
  dataset.py         20 Newsgroups 로드와 범용 입력 검증
  artifacts.py       NumPy 배열, JSON, 모델 저장·로드
  cli.py             build와 search 명령
tests/
  작은 고정 말뭉치를 사용한 단위·통합 테스트
artifacts/
  실행으로 재생성되는 로그, 모델, 검색 인덱스, 이미지
```

외부 데이터셋에 이식할 때는 `texts: Sequence[str]`, `labels: Sequence[int | str]`, 두 개 이상의 라벨, 결측치 없는 500개 이상의 문서를 입력 계약으로 사용한다. 언어가 바뀌면 전처리기만 교체하고 벡터화·검색·분류 인터페이스는 유지한다.

## 4. 데이터 흐름

1. `fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))`로 18,846개 문서를 읽는다.
2. 공백 문서와 입력 형식을 검사하고 전체 문서 수 및 카테고리 분포를 기록한다.
3. 전체 데이터를 계층화 8:2로 분할한다.
4. 학습 텍스트만 전처리해 vocabulary와 IDF를 적합한다.
5. 학습·테스트 텍스트를 같은 vocabulary 공간의 사용자 정의 희소 TF-IDF로 변환한다.
6. 동일한 전처리·설정으로 sklearn 결과를 만들고 사용자 정의 결과와 비교한다.
7. 학습 행렬을 배치로 공급해 선형 SVM을 학습하고 테스트 지표를 산출한다.
8. 전체 18,846개 문서를 학습 vocabulary와 IDF로 변환해 검색 인덱스를 만든다.
9. 검색 쿼리를 같은 공간으로 변환하고 정규화된 희소 벡터의 내적으로 코사인 유사도를 계산한다.
10. 결과와 메타데이터를 저장하고 README에 실제 실행 수치를 반영한다.

## 5. 검색 설계

문서와 쿼리는 L2 정규화되어 있으므로 코사인 유사도는 내적과 같다.

```text
cosine(q, d) = (q · d) / (||q|| * ||d||) = q · d
```

희소 내적은 두 행의 정렬된 열 인덱스 교집합에 해당하는 값만 곱한다. 생략된 원소는 모두 0이므로 밀집 내적과 결과가 같다. 모든 문서의 점수를 NumPy 배열에 기록하고 안정적인 내림차순 정렬로 Top-k를 선택한다.

반환 필드는 `score`, `doc_id`, `label`, `text_snippet`이다. 빈 쿼리 또는 vocabulary에 없는 단어만 포함한 쿼리는 예외 메시지를 출력하고 성공 결과처럼 위장하지 않는다.

CLI는 다음 두 흐름을 제공한다.

```bash
python main.py build
python main.py --query "space shuttle orbit" --topk 5
```

검색 명령에서 산출물이 없으면 먼저 `build`를 실행하라는 구체적인 안내를 제공한다.

## 6. 전처리 기준

- 영문 소문자화
- 영문자 중심 정규식 토큰화
- 숫자, 이메일 주소, URL, 구두점 및 한 글자 토큰 제거
- 프로젝트에 명시적으로 고정한 영어 불용어 제거
- stemming과 동의어 치환은 적용하지 않음

헤더·푸터·인용문은 카테고리 이름이나 작성자 정보에 대한 과적합을 줄이기 위해 데이터 로더 단계에서 제거한다. stemming을 제외하는 이유는 직접 구현 범위를 전처리와 TF-IDF 원리에 집중시키고, 변형된 토큰이 오분류 분석을 어렵게 하지 않기 위해서다. 이 선택의 한계는 README에서 분석한다.

## 7. 오류 처리와 재현성

- 데이터 다운로드 실패: 네트워크와 캐시 위치를 포함한 실행 안내 제공
- 문서/라벨 길이 불일치, 결측치, 단일 라벨: 즉시 명확한 예외 발생
- 비어 있는 vocabulary: 전처리 기준을 확인하라는 오류 발생
- 0벡터 문서: TF-IDF 행은 유지하고 정규화 시 0으로 나누지 않음
- 빈/OOV 검색 쿼리: 검색 불가 원인을 명시
- 저장 산출물의 버전·shape 불일치: 재빌드 안내

Python과 패키지 버전은 `requirements.txt`에 고정 가능한 범위로 명시한다. 데이터 분할, 분류기, 배치 순서에 동일한 seed를 사용한다. 생성 산출물에는 설정과 실행 시각이 아닌 재현에 필요한 파라미터를 기록한다.

## 8. 검증 전략

구현은 작은 고정 말뭉치의 실패 테스트부터 작성한다.

- 전처리: 대소문자, 특수문자, 불용어, 빈 문서
- 희소 행렬: `indptr`, 열 정렬, 행 복원, 배치 복원
- TF: raw count 단계값
- IDF: smoothing 수식의 수기 계산값
- TF-IDF: 정규화 전후 단계값
- sklearn 검증: 최대 절대 오차 `<= 1e-6`
- 희소성: `nnz`, 밀도, 메모리 계산
- 검색: 희소 내적과 밀집 코사인 기준값의 일치, Top-k 순서
- 분류: 고정 seed, 예측 shape, 지표 범위
- CLI: 잘못된 인자, 미구축 상태, 검색 결과 형식

전체 데이터 실행에서는 다음 증거를 생성한다.

- `tfidf_sklearn_validation.json`: sklearn 설정, shape, 최대·평균 오차, PASS
- `search_index_statistics.json`: `nnz`, 밀도, 희소율, 밀집/희소 바이트
- `classification_metrics.json`: 분할 수, 모델 설정, Accuracy, macro F1
- `classification_confusion_matrix.png`: 20개 클래스 혼동 행렬
- `classification_error_examples.json`: 실제값, 예측값, 스니펫을 포함한 5건 이상
- `search_result_examples.json`: 대표 쿼리와 Top-5 결과

README의 수치는 이 산출물에서 가져오며 예시 수치를 실제 결과처럼 작성하지 않는다.

## 9. README 평가 대응

README는 필수 3단 구성을 사용한다.

1. 구현 요약: 전처리, 수식, 희소 행렬 구조, 검색, 분류
2. 검증/실험 결과: 단계별 TF/IDF/TF-IDF, sklearn 오차, 밀도와 메모리, 검색 예시, Accuracy/F1, 혼동 행렬, 오분류 5건 이상
3. 한계와 개선 방향: 어순·문맥·다의어, 전처리 손실, 오분류 원인, 문맥 임베딩 비교 계획, 100만 건 확장 시 인덱싱·샤딩·ANN 방안과 정량 평가 계획

별도로 설치, 전체 빌드, 검색 CLI, 테스트 실행법과 다른 데이터셋 입력 조건을 명시한다. 코사인 유사도와 유클리드 거리의 차이, 행·열 및 vocabulary 매핑도 수식과 작은 예제로 설명한다.

## 10. 제외 범위

- 문서 또는 카테고리 샘플링
- `max_features`를 이용한 임의의 어휘 절단
- BM25, ANN, 웹 서버 또는 UI 구현
- 형태소 분석, stemming, 문맥 임베딩 구현
- 전체 TF-IDF 밀집 `memmap` 생성

이 항목들은 필수 평가 범위를 넓히거나 실행 부담을 키우므로 구현하지 않고, 필요한 경우 README의 개선 방향에서만 다룬다.
