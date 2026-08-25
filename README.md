# 원하는 문서를 똑똑하게 찾아주는 검색기

20 Newsgroups의 **18,846개 문서와 20개 카테고리를 전부 사용**해 전처리, TF-IDF, 코사인 검색, 선형 SVM 분류를 구현한 프로젝트다. 문서나 vocabulary를 샘플링하지 않았으며, TF-IDF 핵심 계산과 검색 유사도는 NumPy로 직접 구현했다.

## 실행 방법

Python 3.10 이상이 필요하다. 첫 build에서는 Scikit-learn이 20 Newsgroups 원본을 내려받으므로 네트워크 연결이 필요하다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

# 전체 18,846개 문서로 검증·분류·검색 인덱스 생성
python main.py build

# 코사인 유사도 Top-5 검색
python main.py --query "space shuttle orbit" --topk 5

# 단위·통합 테스트
python -m pytest -q
```

`build`는 재사용할 검색 인덱스를 `artifacts/runtime/`에 만들고 평가 근거를 `artifacts/reports/`에 기록한다. 검색 인덱스는 크기 때문에 Git에서 제외되며, 검색 전에 각 환경에서 한 번 생성해야 한다.

## 1. 구현 요약

### 데이터셋 선정과 분할

20 Newsgroups는 이진 감성 데이터인 NSMC·IMDB보다 주제 검색을 평가하기 좋고, 4개 범주의 AG News보다 카테고리가 다양하다. 서로 다른 20개 주제는 TF-IDF가 주제별 핵심 단어를 구분하는지, 코사인 유사도가 관련 문서를 상위에 놓는지, BoW가 비슷한 주제를 어디서 혼동하는지 한 데이터셋으로 확인하기 적합하다.

- 전체 문서: 18,846개
- 카테고리: 20개
- 학습: 15,076개
- 테스트: 3,770개
- 분할: stratified 8:2, `random_state=42`
- 누수 방지: vocabulary와 IDF는 학습 문서에만 `fit`; 테스트와 검색 문서는 같은 공간으로 `transform`
- 메타데이터 과적합 방지: loader에서 headers, footers, quotes 제거

원본 말뭉치는 저장소에 재배포하지 않고 Scikit-learn loader로 받는다. 공개 벤치마크의 본문에서 식별 위험과 과적합 가능성이 큰 헤더·서명·인용문을 제거했으며, 저장소에는 평가에 필요한 짧은 스니펫만 남긴다.

### 전처리 파이프라인

`EnglishPreprocessor`가 모든 학습 문서와 검색 쿼리에 같은 기준을 적용한다.

1. 소문자로 변환한다.
2. `[a-z]+(?:'[a-z]+)?` 패턴으로 영문 단어를 토큰화한다.
3. 숫자, URL·이메일의 기호 부분, 구두점과 한 글자 토큰을 제거한다.
4. 코드에 고정한 영어 불용어 목록을 제거한다.
5. stemming과 동의어 치환은 하지 않는다.

특수문자와 숫자는 주제 의미보다 식별자·서식 노이즈가 되는 경우가 많아 제외했다. 불용어는 여러 카테고리에 반복되는 기능어가 높은 빈도를 차지하지 않도록 제거했다. 반면 stemming은 원형 복원 오류가 오분류 해석을 어렵게 할 수 있어 제외했다. 빈 문서는 삭제해 전체 건수를 바꾸지 않고 0벡터로 유지한다.

### TF → IDF → TF-IDF 직접 구현

학습 문서 수를 (N), 문서 (d)에서 단어 (t)의 출현 횟수를 (count(t,d)), 단어가 나타난 문서 수를 (df(t))라고 정의했다.

```text
TF(t, d)       = count(t, d)
IDF(t)         = log((1 + N) / (1 + df(t))) + 1
TF-IDF(t, d)   = TF(t, d) × IDF(t)
L2 정규화       = TF-IDF(d) / ||TF-IDF(d)||₂
```

TF는 문서 내부 중요도를, IDF는 여러 문서에 흔한 단어의 영향 감소를 담당한다. 분자·분모에 1을 더하는 smoothing은 문서 빈도가 낮은 단어의 값이 지나치게 커지는 것을 완화하며, 마지막 `+1`은 sklearn 정의와 일치시킨다.

직접 구현 단계는 서로 분리되어 있다.

- `Counter`로 문서별 raw TF를 만든다.
- 각 문서에서 한 번씩만 열 번호를 세어 DF를 만든다.
- NumPy 로그 연산으로 IDF 배열을 만든다.
- `data *= idf[indices]`로 TF-IDF를 계산한다.
- 문서 행별 L2 norm으로 정규화한다.

실제 학습 문서 0번의 일부 중간값은 다음과 같다.

| 단어 | 열 | TF | IDF | 정규화 전 TF-IDF | 정규화 후 |
|---|---:|---:|---:|---:|---:|
| accomplish | 454 | 1 | 6.957364 | 6.957364 | 0.225654 |
| advantages | 1,094 | 1 | 6.729105 | 6.729105 | 0.218250 |
| ago | 1,493 | 1 | 4.172036 | 4.172036 | 0.135315 |
| answer | 3,020 | 1 | 4.359434 | 4.359434 | 0.141393 |
| application | 3,456 | 1 | 4.900614 | 4.900614 | 0.158945 |

전체 단계값은 `artifacts/reports/stage_example.json`에서 확인할 수 있다.

### 희소 행렬을 선택한 수학적·실측 근거

TF-IDF 행렬의 shape은 `문서 수 × vocabulary 크기`다. 행은 문서 ID, 열은 고정된 단어 ID이며, 쿼리도 반드시 같은 vocabulary 열에 배치되어야 내적이 의미를 갖는다.

문서에 단어가 없으면 `TF=0`이므로 `TF-IDF=0`이고, L2 정규화 이후에도 0이다. 따라서 0을 생략한 표현은 근사나 특징 제거가 아니라 밀집 행렬과 수학적으로 동일하다. 직접 구현한 `SparseMatrix`는 다음 NumPy 배열만 보관한다.

- `data`: 0이 아닌 `float64` TF-IDF 값
- `indices`: 값에 대응하는 vocabulary 열 번호
- `indptr`: 각 문서 행의 시작 위치

```text
density      = nnz / (N × V)
sparsity     = 1 - density
dense bytes  = N × V × 8
sparse bytes = data.nbytes + indices.nbytes + indptr.nbytes
```

전체 검색 행렬의 실제 측정 결과다.

| 항목 | 측정값 |
|---|---:|
| Shape | 18,846 × 89,304 |
| 0이 아닌 원소(`nnz`) | 1,253,490 |
| 밀도 | 0.074478% |
| 희소율 | 99.925522% |
| 예상 밀집 `float64` | 13,464,185,472 bytes (약 12.54 GiB) |
| 직접 구현 희소 배열 | 15,117,268 bytes (약 14.42 MiB) |
| 저장 효율 | 약 890.65배 |

`float64` 값과 `int32` 열 번호를 쓸 때 행 포인터를 제외한 희소 표현의 손익분기 밀도는 대략 `8/(8+4)=66.7%`다. 실제 밀도는 0.075% 미만이므로 희소 표현의 근거가 충분하다.

분류 단계에서만 같은 NumPy `data/indices/indptr` 버퍼를 Scikit-learn 의존성이 제공하는 SciPy CSR 컨테이너로 연결해 sklearn에 전달한다. 이 경계는 TF-IDF를 다시 계산하지 않는다. TF, IDF, 정규화와 검색 계산은 모두 프로젝트의 NumPy 구현을 사용한다.

### 코사인 유사도 검색

코사인 유사도는 다음과 같다.

```text
cosine(q, d) = (q · d) / (||q||₂ × ||d||₂)
```

문서와 쿼리를 이미 L2 정규화했으므로 검색 시 `cosine(q,d)=q·d`다. 직접 구현한 두 포인터 알고리즘이 쿼리와 문서에서 0이 아닌 열 번호의 교집합만 찾아 곱한다.

```text
q · d = Σ(i ∈ nz(q) ∩ nz(d)) q[i] × d[i]
```

생략된 모든 항은 0이므로 밀집 내적과 결과가 같다. 전체 문서를 정확히 스캔해 점수 배열을 만들고, 점수 내림차순·문서 ID 오름차순으로 Top-k를 반환한다. 결과는 `score`, `doc_id`, `label`, `text_snippet`을 포함한다.

유클리드 거리는 벡터의 절대 크기에 영향을 받지만 코사인은 방향을 비교해 문서 길이 영향을 줄인다. L2 정규화된 두 벡터에서는 `||q-d||²=2(1-cosine(q,d))`라 순위가 같지만, 정규화하지 않은 TF-IDF에서는 긴 문서가 유클리드 거리에 불리할 수 있다. 반대로 코사인은 단어 비율만 비슷한 매우 짧은 문서를 과대평가할 수 있다.

### 분류 시스템과 모듈 구조

분류기는 `SGDClassifier(loss="hinge")`다. 결정 함수 `w·x+b`와 hinge loss를 최적화하는 선형 SVM이며, 희소 행을 128개씩 `partial_fit`한다. 6 epochs, `random_state=42`로 고정했다.

| 모듈 | 책임 |
|---|---|
| `preprocessing.py` | 정제, 토큰화, 불용어 제거 |
| `dataset.py` | 데이터 로드와 범용 입력 검증 |
| `sparse_matrix.py` | NumPy `data/indices/indptr`, 행·배치 변환, 메모리 통계 |
| `tfidf.py` | vocabulary, TF, IDF, 정규화, query transform |
| `validation.py` | sklearn 동등성 검증과 단계값 생성 |
| `search.py` | 희소 내적, 코사인 점수, Top-k |
| `classification.py` | CSR 입력 연결, 선형 SVM, 지표, 혼동 행렬 |
| `artifacts.py` | 검색 인덱스의 버전 기반 저장·복원 |
| `pipeline.py` | 전체 데이터 분할·학습·평가·산출물 생성 |
| `cli.py` | build와 검색 CLI |

## 2. 검증/실험 결과

### TF-IDF 구현 검증

학습 행렬 전체를 밀집 변환하지 않고, 직접 구현과 sklearn CSR의 행별 열 번호와 값을 비교했다.

```text
shape: (15,076, 89,304)
smooth_idf=True
sublinear_tf=False
norm='l2'
use_idf=True
dtype=float64
최대 절대 오차: 7.549516567451064e-15
평균 절대 오차: 8.379477537989127e-21
허용 오차: 1e-6
결과: PASS
```

`sublinear_tf=False`이므로 TF는 `1+log(count)`가 아닌 raw count다. `smooth_idf=True`는 IDF 수식의 분자·분모에 1을 더한다. `norm='l2'`는 각 문서 벡터 길이를 1로 만든다. 이 설정을 직접 구현과 고정해 계산 정의 차이로 인한 가짜 오차를 피했다.

### 실제 검색 결과

쿼리 `space shuttle orbit`의 Top-5 결과는 모두 `sci.space`였다.

| 순위 | 점수 | 문서 ID | 카테고리 | 스니펫 |
|---:|---:|---:|---|---|
| 1 | 0.411651 | 4,672 | sci.space | SPACE SHUTTLE ANSWERS, LAUNCH SCHEDULES... |
| 2 | 0.370417 | 5,788 | sci.space | It flies. It lands. It gets rebuilt... |
| 3 | 0.368532 | 4,389 | sci.space | CONTROVERSIAL QUESTIONS... |
| 4 | 0.355928 | 12,888 | sci.space | notice posted weekly in sci.space... |
| 5 | 0.353530 | 10,571 | sci.space | NETWORK RESOURCES OVERVIEW... |

`baseball pitcher season`과 `computer graphics image`도 각각 Top-5가 모두 `rec.sport.baseball`, `comp.graphics`였다. 전체 원문과 점수는 `artifacts/reports/search_examples.json`에 기록했다.

### 분류 성능

```text
모델: SGDClassifier(loss='hinge') 선형 SVM
학습/테스트: 15,076 / 3,770
Accuracy: 0.759416
Macro F1: 0.747980
오분류: 907건
```

![20개 카테고리 혼동 행렬](artifacts/reports/confusion_matrix.png)

대각선이 전반적으로 강하지만 `alt.atheism`·`soc.religion.christian`·`talk.religion.misc`, 그리고 정치 하위 카테고리 사이의 혼동이 상대적으로 보인다. 이 범주들은 같은 주제 단어를 공유하면서 주장 방향과 맥락이 달라 BoW에 어렵다.

### 오분류 사례 분석

아래 사례는 고정된 테스트 분할의 실제 결과다.

1. **문서 204 — comp.graphics → talk.politics.misc**  
   본문이 “CRT”, “register state”를 묻는 한 문장뿐이다. 그래픽 분야를 확정할 단어가 부족하고 `state` 같은 다의어가 다른 주제에서도 나타난다. 짧은 문서 최소 길이 분석이나 character n-gram을 추가하면 개선 가능하다.

2. **문서 8,510 — talk.politics.mideast → rec.motorcycles**  
   headers·footers·quotes 제거 후 본문이 빈 문자열이 되어 0벡터가 됐다. 선형 모델은 단어 근거 없이 절편으로 클래스를 선택한다. 빈 문서를 별도 표시하거나 원본 정제 정책을 개선해야 한다.

3. **문서 14,558 — alt.atheism → soc.religion.christian**  
   `moral`, `churches`, `sin`처럼 기독교 문서에서 흔한 단어가 포함되지만 글쓴이의 비판적 입장은 단어 집합만으로 드러나지 않는다. BoW가 관점과 부정 관계를 무시한 사례로, bigram 또는 문맥 임베딩이 필요하다.

4. **문서 12,024 — comp.sys.mac.hardware → comp.sys.ibm.pc.hardware**  
   modem, fax, data, 14.4k처럼 두 하드웨어 카테고리가 공유하는 단어가 중심이다. `Centris 650`이라는 Mac 단서는 희귀해 공통 하드웨어 단어보다 영향이 작았다. 개체·제품명 보존과 character n-gram이 도움이 된다.

5. **문서 12,573 — sci.crypt → talk.politics.guns**  
   `peaceful`, `blood`, `revolution` 같은 정치·폭력 관련 단어가 표면에 많고 암호학 맥락은 앞뒤 대화에 의존한다. 인용문 제거와 BoW의 문맥 단절이 함께 작용했다. 문장 수준 임베딩이나 대화 문맥 보존 실험이 필요하다.

6. **문서 15,922 — sci.space → talk.politics.mideast**  
   실제 본문은 시대별 Native American culture를 논하며 우주 관련 표면 단어가 거의 없다. 게시 카테고리와 한 게시물의 국소 주제가 다를 수 있다는 라벨 노이즈 사례다. 문서 단위 주제 분포를 모델링하거나 저신뢰 예측을 별도 검토해야 한다.

전체 수집 사례 20건은 `artifacts/reports/misclassifications.json`에 있다.

## 3. 한계와 개선 방향

### BoW의 근본적 한계

TF-IDF는 각 열을 독립 단어로 취급한다.

- **어순 무시:** “dog bites man”과 “man bites dog”는 같은 단어 빈도라 동일 벡터가 된다. 검색 결과와 분류가 사건 주체를 구분하지 못한다.
- **부정·관점 무시:** “this policy is good”과 “this policy is not good”은 불용어에서 `not`까지 제거하면 거의 같은 특징이 된다.
- **다의어:** `apple`은 과일과 회사 문맥에서 같은 열 하나다. 한국어의 `사과`도 과일과 사죄를 같은 feature로 처리한다.
- **동의어 분리:** `car`와 `automobile`은 의미가 가까워도 서로 다른 열이라 유사도가 낮아진다.
- **희귀 표현과 짧은 문서:** 주제 단서가 적으면 특정 단어 하나나 모델 절편의 영향이 커진다.

후속 실험에서는 word/character n-gram 기준선을 먼저 추가하고, 문맥 임베딩 또는 BERT 계열 문장 벡터와 비교한다. 동일한 8:2 분할에서 Accuracy와 macro F1을 비교하고, 카테고리가 같은 문서를 관련 문서로 정의해 검색 mAP@5·Recall@5도 측정한다. 임베딩 생성 시간, 인덱스 크기, 쿼리 p95 latency까지 함께 기록해야 정확도 향상이 운영 비용을 정당화하는지 판단할 수 있다.

### 검색 평가의 한계

현재 검색 예시는 정성적으로 관련 카테고리가 상위에 오는지 확인했지만, 사람의 relevance judgment나 mAP을 아직 계산하지 않았다. 같은 카테고리가 항상 같은 검색 의도를 뜻하지도 않는다. 다음 버전에서는 쿼리 세트를 고정하고 카테고리 기반 약식 지표와 수동 relevance label 기반 mAP@5를 함께 측정한다.

### 전처리 개선

숫자 제거는 modem 속도 `14.4k`나 제품 모델명을 손실시킬 수 있고, headers·quotes 제거는 누수를 줄이는 대신 문맥 전체를 없애 빈 문서를 만들기도 한다. 다음 실험은 현재 기준, 숫자·제품명 보존, character n-gram, stemming 적용을 한 요소씩 바꾸고 macro F1과 빈 벡터 수를 비교한다.

### 100만 건 규모 확장

현재 정확 검색은 모든 문서를 스캔하므로 문서 수 (N), 문서당 평균 비제로 원소 수 (k)에 대해 대략 `O(N × (nnz_query + k))`다. 100만 건에서는 다음 병목이 생긴다.

- 전체 스캔으로 쿼리 응답시간이 선형 증가한다.
- vocabulary와 `nnz` 증가로 인덱스 메모리·디스크가 커진다.
- 단일 프로세스의 전처리와 인덱스 빌드 시간이 길어진다.
- 검색 인덱스 갱신과 모델 재학습이 일괄 작업 병목이 된다.

개선 순서는 다음과 같다.

1. 단어 → `(doc_id, weight)` posting list의 역색인을 만들어 쿼리 단어가 있는 문서만 평가한다.
2. 카테고리·기간 등으로 샤딩하고, 자주 쓰는 posting list를 메모리에 캐시한다.
3. 문맥 임베딩을 채택할 경우 HNSW/IVF 같은 ANN으로 후보를 찾고 정확 점수로 재정렬한다.
4. 전처리와 인덱스 생성을 문서 청크별 병렬 작업으로 만들고 증분 갱신을 지원한다.

개선 전후에는 인덱스 크기, build 처리량, 평균/p95 검색 latency, mAP@5와 Recall@5를 같은 하드웨어에서 측정한다. ANN은 속도를 얻는 대신 recall이 낮아질 수 있으므로 정확 검색을 기준선으로 유지한다.

### 다른 데이터셋으로 이식

벡터화 이후 모듈은 언어와 무관하며 다음 입력 계약을 사용한다.

```python
texts: Sequence[str]
labels: Sequence[int | str]
```

- 문서와 라벨 길이가 같아야 한다.
- 결측 텍스트가 없어야 한다.
- 최소 500개 문서와 두 개 이상의 라벨이 필요하다.
- 개인정보가 제거된 공개 데이터 사용을 우선한다.
- 한국어는 `EnglishPreprocessor`를 형태소 분석기 기반 구현으로 교체해야 한다. 공백 토큰화는 조사·어미가 붙은 형태를 서로 다른 단어로 만들어 vocabulary를 키우고 유사도·분류 성능을 낮출 수 있다.

`NumpyTfidfVectorizer`, `DocumentSearch`, 분류·검증 모듈의 인터페이스는 그대로 유지할 수 있다.

## 재현 산출물

| 파일 | 내용 |
|---|---|
| `tfidf_validation.json` | sklearn 설정, shape, 최대·평균 오차와 PASS |
| `matrix_stats.json` | `nnz`, 밀도, 희소율, 밀집·희소 byte 비교 |
| `metrics.json` | 분할, 모델 설정, Accuracy, macro F1 |
| `stage_example.json` | TF → IDF → TF-IDF 중간값 |
| `search_examples.json` | 세 쿼리의 실제 Top-5 |
| `misclassifications.json` | 실제값·예측값·스니펫 20건 |
| `confusion_matrix.png` | 20개 클래스 혼동 행렬 |
