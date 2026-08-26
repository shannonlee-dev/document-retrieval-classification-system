# 원하는 문서를 똑똑하게 찾아주는 검색기

20 Newsgroups의 세 주제에 deterministic structured-PII redaction과 artifact 최소화를 적용한 뒤 NumPy TF-IDF와 코사인 검색 및 선형 SVM 분류를 재현하는 privacy-conscious 프로젝트다.

## 실행 방법

Python 3.10 이상이 필요하다. 첫 build는 scikit-learn loader가 원본 데이터셋을 내려받으므로 네트워크 연결이 필요할 수 있다.

### uv 사용

```bash
uv sync --extra dev
uv run python main.py build
uv run python main.py --query "space shuttle orbit" --topk 5
uv run pytest -q
```

### venv와 pip 사용

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python main.py build
python main.py --query "space shuttle orbit" --topk 5
python -m pytest -q
```

`build`는 `artifacts/runtime/`의 검색 인덱스와 `artifacts/reports/`의 평가 자료를 같은 경로에 덮어쓴다. 변경 전 runtime metadata는 호환 로딩하지 않으며, 검색 시 재빌드 안내 오류를 낸다.

## 데이터와 개인정보 경계

scikit-learn의 `fetch_20newsgroups(subset="all")`에서 다음 세 카테고리만 불러온다.

- `comp.graphics`
- `rec.sport.baseball`
- `sci.space`

### 왜 세 카테고리만 선택했는가

이 세 카테고리는 미션 예시와 같고 서로 구분되는 기술·스포츠·과학 주제라 다중 분류 결과를 설명하기 쉽다. 다른 카테고리가 모두 위험하다는 뜻은 아니며, 카테고리 선택 자체를 개인정보 보호 수단으로 간주하지 않는다. 세 카테고리 안의 모든 문서에도 같은 privacy pipeline을 적용한다.

### Privacy pipeline과 데이터 경계

loader는 headers, footers, quotes를 제거한 뒤 이메일, 명확한 전화번호, URL과 유효한 IPv4·IPv6를 deterministic하게 redaction한다. redaction과 영문 정규화 뒤 빈 문서만 제외한다. 사람 이름·주소 같은 자유형 엔터티를 추측하는 detector나 의심 문서 전체 삭제는 적용하지 않는다.

정제된 전체 문서는 실행 중 TF-IDF fit/transform, 분류와 검색 행렬 생성에만 사용하고 저장하지 않는다. runtime metadata, 검색 결과와 오분류 보고서에는 structured-PII sanitization을 다시 적용하고 검증한 최대 240자의 snippet만 기록한다. 공개 검색 문서 ID는 filtering 전 loader 행 번호를 유지한다.

이 설계는 “PII-free”를 보증하지 않는다. 대신 상대적으로 저위험인 세 카테고리 선택, metadata 제거, 확실한 structured identifier redaction, full-text 비저장과 bounded snippet으로 잔여 노출 범위를 줄인다.

### 실제 privacy 처리 결과

| 범위 | raw | retained | excluded | 유지율 |
|---|---:|---:|---:|---:|
| 전체 | 2,954 | 2,863 | 91 | 96.92% |
| `comp.graphics` | 973 | 953 | 20 | 97.94% |
| `rec.sport.baseball` | 994 | 956 | 38 | 96.18% |
| `sci.space` | 987 | 954 | 33 | 96.66% |

카테고리 유지율의 최대 차이는 약 1.76%p다. 현재 측정에서는 sanitization 후 빈 문서 제외 때문에 한 클래스만 크게 무너지는 현상이 없다.

| 제외 사유 | 문서 수 |
|---|---:|
| sanitization 후 빈 문서 | 91 |

| redaction 종류 | 횟수 |
|---|---:|
| email | 1,237 |
| phone | 566 |
| URL | 1 |
| IPv4 | 485 |
| IPv6 | 8 |

데이터 출처, 귀속, 라이선스 및 파생 정책은 [DATASET_LICENSE.md](DATASET_LICENSE.md)에 기록했다.

## 전처리와 TF-IDF

`EnglishPreprocessor`는 모든 문서와 검색 쿼리에 같은 순서를 적용한다.

1. 소문자로 변환한다.
2. 영문자와 공백 외 문자를 공백으로 치환한다.
3. whitespace로 분할한다.
4. 한 글자 토큰과 고정 불용어를 제거한다.

동의어 치환은 사용하지 않는다. 외부 어휘 자원에 새로운 의미 가정을 의존하게 되고, 문맥에 맞지 않는 치환이 생길 수 있으며, 직접 구현과 scikit-learn 검증의 분석 경계도 불필요하게 달라지기 때문이다.

TF-IDF는 다음 정의를 NumPy 배열과 자체 `SparseMatrix`로 계산한다.

```text
TF(t, d)       = count(t, d)
IDF(t)         = log((1 + N) / (1 + df(t))) + 1
TF-IDF(t, d)   = TF(t, d) × IDF(t)
L2 정규화       = TF-IDF(d) / ||TF-IDF(d)||₂
```

학습 문서에서만 vocabulary와 IDF를 fit하고 테스트 및 전체 검색 문서는 같은 공간으로 transform한다. scikit-learn `TfidfVectorizer`와의 최대 절대 오차 허용치는 `1e-6`이다. 검증 시 직접 구현의 raw-count TF, smoothed IDF, L2 정규화와 맞추기 위해 다음 설정을 고정한다.

```text
smooth_idf=True
sublinear_tf=False
norm="l2"
use_idf=True
dtype=float64
```

## 검색과 분류

문서와 쿼리 벡터는 L2 정규화되어 있으므로 코사인 유사도는 내적과 같다. `sparse_dot`은 `np.intersect1d(..., return_indices=True)`로 공통 열의 위치를 찾고 `np.dot`으로 대응 TF-IDF 값을 계산한다. 동점은 공개 source document ID 오름차순으로 정렬한다.

분류는 `SGDClassifier(loss="hinge")`를 사용한다. 프로젝트와 테스트 코드는 SciPy를 직접 import하지 않으며, 학습과 예측 때 자체 희소 행렬의 행을 `batch_size` 이하 NumPy 밀집 배열로만 변환한다. 전체 말뭉치를 한 번에 밀집화하지 않는다.

## 재현 결과

고정 설정은 stratified 8:2 분할, `random_state=42`, 세 카테고리다. 현재 저장된 보고서는 다음 결과를 기록한다.

| 항목 | 재빌드 결과 |
|---|---:|
| 정제 후 문서 | 2,863 |
| 학습 / 테스트 문서 | 2,290 / 573 |
| vocabulary | 21,662 |
| 전체 행렬 shape | 2,863 × 21,662 |
| `nnz` | 178,950 |
| 밀도 / 희소율 | 0.288544% / 99.711456% |
| 밀집 / 자체 희소 저장 크기 | 496,146,448 / 2,158,856 bytes |
| 밀집 대비 저장 효율 | 229.8191배 |
| TF-IDF 최대 절대 오차 | 1.6653345369377348e-15 |
| Accuracy | 0.9040139616055847 |
| macro F1 | 0.9030980112997345 |
| 전체 오분류 | 55 |

기존 68-term allowlist 결과와 비교하면 retained 문서는 2,048개에서 2,863개로, vocabulary는 68개에서 21,662개로, `nnz`는 6,260에서 178,950으로 증가했다. 정확도는 약 94.15%에서 90.40%로 낮아졌지만, 주제 정답을 암시하도록 수동 선택한 68개 단어가 아니라 실제 문장 어휘를 사용하는 더 현실적인 실험이다.

`space shuttle orbit`의 Top-5도 모두 `sci.space`였다.

| 순위 | 점수 | source document ID | 카테고리 |
|---:|---:|---:|---|
| 1 | 0.340539 | 2,150 | `sci.space` |
| 2 | 0.324998 | 42 | `sci.space` |
| 3 | 0.301737 | 566 | `sci.space` |
| 4 | 0.296000 | 897 | `sci.space` |
| 5 | 0.290371 | 2,710 | `sci.space` |

![세 카테고리 혼동 행렬](artifacts/reports/confusion_matrix.png)

정확한 원시 값과 안전 스니펫은 build가 생성한 다음 파일을 기준으로 한다.

| 파일 | 내용 |
|---|---|
| `artifacts/reports/metrics.json` | 문서·분할 수, 모델 설정, Accuracy, macro F1 |
| `artifacts/reports/tfidf_validation.json` | scikit-learn 대조 설정과 오차 |
| `artifacts/reports/matrix_stats.json` | shape, `nnz`, 밀도, 저장 byte |
| `artifacts/reports/privacy_report.json` | redaction·제외·카테고리별 유지 통계 |
| `artifacts/reports/stage_example.json` | TF → IDF → TF-IDF 중간값 |
| `artifacts/reports/search_examples.json` | 세 쿼리의 안전한 Top-5 스니펫 |
| `artifacts/reports/misclassifications.json` | 안전한 오분류 스니펫 최대 20건 |
| `artifacts/reports/confusion_matrix.png` | 세 클래스 혼동 행렬 |

## 과거 20-Category 기준선

privacy-conscious redesign 이전에는 전체 20 Newsgroups 말뭉치(20개 카테고리, 문서 18,846개)로도 실험을 수행했다. 아래 값은 공개 benchmark에 대한 **historical pre-privacy-redesign baseline**으로 보존하는 aggregate metrics이며, 현재 privacy-conscious pipeline의 공식 성능이 아니다.

### 분류 품질

| 실험 | 카테고리 | 문서 | Accuracy | Macro-F1 | 목적 |
|---|---:|---:|---:|---:|---|
| 과거 전체 말뭉치 기준선 | 20 | 18,846 | 75.94% | 74.80% | privacy redesign 이전 engineering baseline |
| 현재 privacy-conscious 실험 | 3 | 2,863 retained | 90.40% | 90.31% | 현재 지원하는 실험 |

이 점수들은 직접 비교할 수 없다. historical 결과는 structured-PII redaction 및 artifact 최소화 정책보다 앞선 전체 말뭉치 실험이고, 현재 결과는 서로 다른 데이터 범위와 전처리·privacy 정책을 적용한 3-category 실험이다.

historical 분류 결과는 Accuracy 75.94%, Macro-F1 74.80%, 오분류 907 / 3,770 test documents이며, 혼동 행렬은 20 × 20이다. 이 수치는 과거 engineering baseline을 참고하기 위한 것이며, 현재 privacy-conscious system의 성능으로 해석해서는 안 된다.

### 표현 방식 및 자원 효율

다음 historical TF-IDF 수치는 분류 품질이 아니라 표현 방식과 자원 효율을 설명한다. Accuracy나 Macro-F1의 향상으로 해석하지 않는다.

| 항목 | Historical full-corpus 값 |
|---|---:|
| 전체 행렬 shape | 18,846 × 89,304 |
| `nnz` | 1,253,490 |
| 희소율 | 약 99.925522% |
| 예상 dense `float64` 크기 | 13,464,185,472 bytes |
| 자체 sparse 표현 크기 | 15,117,268 bytes |
| dense 대비 크기 | 약 890.65배 작음 |
| scikit-learn TF-IDF validation 최대 절대 오차 | 약 7.55e-15 |

과거 실험의 전체 세부사항은 commit `dd9e42d261ae8d4a3a876906f77e539aba09e630`에서 확인할 수 있다.

## 한계와 개선 방향

### 오분류 5건 분석

아래 분석은 `misclassifications.json`에 저장된 정제 후 스니펫을 기준으로 한다. 스니펫은 최대 240자로 제한되므로 원문의 전체 문맥을 보존하지 않으며, 이 자체가 짧은 텍스트에서 BoW 분류가 약해지는 이유이기도 하다.

| source document ID | 실제 → 예측 | 관찰한 원인 | BoW 한계와의 연결 |
|---:|---|---|---|
| 1,404 | `comp.graphics` → `sci.space` | `virtual worlds`, `directory`처럼 일반적이고 짧은 표현만 남아 그래픽스 고유 단어가 거의 없다. | 단어의 출현 빈도만으로는 문서의 주제와 대화 맥락을 복원할 수 없다. |
| 15 | `comp.graphics` → `sci.space` | 예산, IBM, 소프트웨어 등 일반 기술 용어가 중심이고 그래픽스 관련 단어가 스니펫에 나타나지 않는다. | 문단의 앞뒤 맥락이나 게시물의 원래 질문을 반영하지 못해 일반 단어의 학습 빈도에 좌우된다. |
| 2,495 | `comp.graphics` → `sci.space` | `ray tracing`은 그래픽스 단서지만 한 번만 등장하고, 나머지는 조언·참고자료 같은 일반 문장이다. | `ray tracing` 같은 복합 개념을 단어 두 개의 독립 빈도로만 처리하므로 구(phrase)의 의미를 충분히 반영하지 못한다. |
| 58 | `rec.sport.baseball` → `sci.space` | 동의·투표 같은 대화 표현뿐이며 야구를 가리키는 단어가 없다. | 짧고 정보량이 적은 문서는 클래스별 TF-IDF 특징이 거의 없어 분류 근거가 약하다. |
| 2,846 | `sci.space` → `rec.sport.baseball` | 배우와 연기에 관한 문장으로, 실제 라벨인 우주 주제를 보여 주는 단어가 없다. | 인용문·주제 이탈·라벨 잡음처럼 문서 내용과 클래스가 어긋난 경우에는 단어 빈도 기반 모델이 원래 카테고리를 추론할 수 없다. |

### 개선 방향

- 단어 bi-gram을 추가해 `ray tracing`처럼 함께 나타날 때 의미가 달라지는 표현을 보존한다.
- 매우 짧거나 어휘가 없는 문서는 낮은 신뢰도로 표시해 재검토 대상으로 분리한다.
- 다음 학습 단계에서는 문맥을 표현하는 임베딩 기반 모델과 원문 단위 라벨 점검을 비교해 BoW 기준선의 한계를 정량적으로 확인한다.

- structured identifier regex는 자유 형식의 사람 이름·주소·건강정보를 탐지하지 않는다. 이 pipeline은 PII-free 보증이 아니며, 실제 배포에는 별도의 데이터 거버넌스 검토가 필요하다.
- BoW는 어순, 부정, 관점 및 다의어를 직접 표현하지 못한다.
- 현재 검색은 모든 문서를 순회하는 정확 검색이므로 문서 수에 따라 지연 시간이 선형 증가한다.
- 카테고리 일치는 정성적 검색 예시일 뿐 사람의 relevance judgment나 mAP 평가를 대신하지 않는다.
