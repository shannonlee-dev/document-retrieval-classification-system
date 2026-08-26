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

### 데이터 윤리 요구를 어떻게 해석했는가

이 프로젝트는 “공개 라이브러리가 제공하므로 개인정보 검토가 필요 없다”고 가정하지 않는다. 공개 여부와 개인정보 위험은 별개의 문제다. 다만 미션이 20 Newsgroups를 사용 가능한 영어 데이터셋으로 직접 제시한다는 점을 고려하면, 데이터 윤리 요구의 취지는 추천 데이터셋 안에 어떤 문자열도 개인정보처럼 보이지 않아야 한다는 절대적 보증보다 출처가 불분명하거나 민감한 데이터를 무심코 사용하지 않고, 식별 위험을 확인해 필요한 범위만 처리·보관하라는 것으로 해석했다.

자유 서술형 문서에서 모든 사람 이름·주소·건강 관련 표현의 부재를 자동으로 증명하는 것은 현실적으로 어렵다. 사람 이름처럼 보이는 단어도 작성자, 공인, 제품명 또는 일반 단어일 수 있고, 건강 관련 단어가 있다는 사실만으로 특정인의 건강정보가 되는 것도 아니다. 반대로 공개 데이터라는 이유만으로 실제 연락처나 작성자 metadata를 그대로 유지하는 것도 적절하지 않다. 그래서 이 프로젝트는 불확실한 문자열을 모두 삭제했다고 주장하는 대신, 직접 식별 가능성이 높은 정보부터 결정론적으로 제거하고 저장되는 텍스트의 양을 제한하는 방식을 선택했다.

| 결정 | 이렇게 한 이유 | 남는 한계와 대응 |
|---|---|---|
| scikit-learn의 20 Newsgroups 사용 | 미션이 직접 추천하고, 출처와 로딩 과정이 공개되어 재현과 귀속이 가능하다. 공개 데이터라는 사실만을 면책 근거로 사용하지는 않는다. | 원문이 자유 서술형이므로 loader 경계에서 모든 문서에 동일한 sanitization을 적용한다. |
| 세 카테고리만 선택 | 검색·분류 실험에 필요한 서로 다른 주제를 확보하면서 처리·검토 범위를 줄일 수 있다. | 카테고리 선택 자체는 개인정보 제거 수단이 아니므로 모든 카테고리에 같은 정책을 적용한다. |
| headers, footers, quotes 제거 | 작성자·서명·연락 경로가 포함되기 쉬운 metadata와 반복 인용문은 주제 분류에 필수적이지 않으면서 노출 범위를 키운다. | 본문에 남은 structured identifier는 다음 redaction 단계에서 다시 처리한다. |
| 이메일·전화번호·URL·유효한 IP를 deterministic redaction | 직접 식별이나 연락에 사용될 가능성이 높고, 정규식과 IP 파서로 처리 기준을 명확하게 재현·검증·집계할 수 있다. | 자유형 이름과 주소까지 완전히 탐지하는 정책은 아니므로 PII-free라고 표현하지 않는다. |
| 자유형 엔터티를 추측해 일괄 삭제하지 않음 | 이름 추정은 오탐과 누락이 모두 발생할 수 있고, 주제 단어를 자의적으로 제거해 TF-IDF와 클래스 분포까지 왜곡할 수 있다. | 식별 가능성이 완전히 사라졌다고 가정하지 않고 full text를 저장하지 않으며 출력 snippet을 제한한다. |
| sanitization 후 빈 문서만 제외 | “의심된다”는 주관적 기준으로 문서를 제거하면 재현성이 떨어지고 특정 클래스가 더 많이 제외되는 선택 편향이 생길 수 있다. | build마다 제외 수와 카테고리별 유지율을 보고해 편향 여부를 확인한다. |
| full text 비저장, 최대 240자 snippet만 기록 | 모델 재현에 필요한 행렬·통계와 검색 결과의 확인 가능성은 유지하면서 원문 재배포와 불필요한 노출을 줄인다. | snippet에도 잔여 자유형 정보가 있을 수 있으므로 저장 직전에 structured sanitization을 다시 적용하고 한계를 명시한다. |

#### 시행착오와 현재 정책을 선택한 근거

현재 정책을 처음부터 전제로 삼은 것은 아니다. commit `56e2c33`에서는 개인정보처럼 보이는 자유 텍스트를 남기지 않기 위해 graphics, baseball, space 분야에서 미리 고른 68개 일반 주제어만 허용했다. 이 방식은 규칙이 단순하고 허용되지 않은 이름·연락처·건강 관련 표현을 함께 제거한다는 장점이 있었지만, 개인정보 처리 규칙과 ML vocabulary를 하나로 묶는 문제가 있었다. 검색·분류에 쓸 수 있는 단어를 개인정보 정책이 미리 결정하므로 모델은 실제 문서가 아니라 사람이 골라 준 클래스별 핵심어만 보게 되었다.

commit `f4fdefd`의 allowlist 실험과 commit `d45cdfe` 이후 현재 실험은 다음 차이를 보였다. 두 실험은 같은 세 카테고리와 8:2 계층 분할을 사용하지만 정제 후 남은 문서 집합과 feature 공간이 다르므로, 점수만으로 우열을 직접 판단하지 않는다.

| 항목 | 68-term allowlist | 현재 structured redaction | 변화의 의미 |
|---|---:|---:|---|
| retained 문서 | 2,048 | 2,863 | 815개, 약 39.8% 더 많은 문서를 실험에 사용한다. |
| vocabulary | 68 | 21,662 | 사람이 고른 주제어가 아니라 정제된 실제 문장 어휘를 학습한다. |
| `nnz` | 6,260 | 178,950 | 문서가 가진 검색·분류 단서를 훨씬 더 많이 보존한다. |
| Accuracy | 94.15% | 90.40% | 약 3.74%p 낮아졌지만, 클래스 정답을 암시하는 수동 어휘 선택 효과를 제거했다. |
| macro F1 | 94.21% | 90.31% | 약 3.90%p 낮아졌으며, 더 크고 어려운 feature 공간에서 측정한 결과다. |

allowlist 검색은 `space shuttle orbit`에 0.92에 가까운 높은 유사도를 냈지만, 결과 snippet도 `mission space orbit ... shuttle`처럼 허용된 주제어의 반복만 남았다. 높은 점수와 낮은 오분류 수는 얻었어도 사용자가 문서를 읽고 관련성을 판단할 문맥이 사라졌고, 새로운 질의어가 68개 목록 밖에 있으면 검색할 수도 없었다. 따라서 이 점수는 개인정보 보호와 모델 성능을 동시에 개선했다는 증거라기보다, 사람이 정답과 가까운 feature를 먼저 선택해 문제를 단순화했을 때 나타난 결과로 해석하는 편이 타당하다.

allowlist 없이 모든 사람 이름과 주소를 제거하는 방법도 검토했지만, 범용 자유형 엔터티 탐지는 이 미션과 별도의 문제가 된다. `Alice Smith`가 개인인지 예시 문구인지, `Jordan`이 사람인지 지명인지, 어떤 건강 관련 문장이 특정 개인의 건강정보인지 판정하려면 NER 또는 비식별화 모델, 별도의 정답 라벨, precision·recall 평가와 오탐·누락 분석이 필요하다. 이를 검증하지 않은 단순 이름 목록이나 대문자 규칙은 PII-free라는 잘못된 확신을 주고, 사람·회사·제품·기술 용어를 함께 삭제해 TF-IDF 실험을 다시 왜곡할 수 있다. 이 때문에 commit `bab416e`의 재설계 기록에서는 자유형 이름·주소 NER를 명시적으로 범위에서 제외하고, 프로젝트의 중심을 개인정보 탐지기가 아니라 NumPy TF-IDF 검색·분류에 유지했다.

결국 현재 정책은 가장 높은 분류 점수나 가장 강한 개인정보 제거 주장 중 하나를 택한 것이 아니다. 직접 식별 가능성이 높고 규칙으로 검증할 수 있는 structured identifier는 제거하되, ML에 필요한 일반 어휘와 문맥은 보존하고, 남는 불확실성은 full-text 비저장·bounded snippet·처리 통계·한계 공개로 관리하는 절충이다. 범위가 더 넓은 개인정보 탐지가 실제 요구사항이 된다면 현재 regex에 추측성 규칙을 덧붙이기보다, 별도의 비식별화 데이터셋과 평가 기준을 가진 독립 단계로 검증하는 것이 맞다.

따라서 이 정책은 개인정보가 절대로 존재하지 않는다는 인증이 아니라, 미션에서 추천한 공개 연구 데이터셋을 교육용 검색·분류 실험에 필요한 범위로 제한하고 식별 가능성이 높은 정보와 불필요한 원문 보관을 줄인 위험 기반 처리다. 운영 서비스나 민감정보 데이터셋에 그대로 적용할 수준의 보증은 아니지만, 데이터 윤리 요구를 무시한 것이 아니라 선택 근거, 처리 범위, 측정 결과와 잔여 한계를 검토 가능한 형태로 남겼다는 데 의미가 있다.

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

기존 68-term allowlist와 현재 정책의 상세 비교 및 변경 근거는 앞의 「시행착오와 현재 정책을 선택한 근거」에 기록했다.

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
