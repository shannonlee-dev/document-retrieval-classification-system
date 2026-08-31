# 원하는 문서를 똑똑하게 찾아주는 검색기

## 프로젝트 소개

20 Newsgroups 문서를 대상으로 **TF-IDF 기반 문서 검색과 주제 분류를 직접 구현하고 검증하는 프로젝트**다.

scikit-learn의 완성된 TF-IDF 구현을 그대로 사용하는 대신, 텍스트를 전처리하고 vocabulary와 IDF를 학습한 뒤 NumPy 배열과 자체 희소 행렬 표현으로 TF-IDF를 계산한다. 분류와 검색은 같은 전처리를 쓰되 목적에 맞는 독립된 TF-IDF 공간을 사용하며, 직접 구현한 결과는 scikit-learn과 수치적으로 비교해 검증한다.

또한 공개 텍스트 데이터라도 불필요한 식별정보와 원문을 그대로 보관하지 않도록 structured-PII redaction과 artifact 최소화를 적용한다. 이 privacy 처리는 검색·분류 알고리즘과 분리된 데이터 경계에서 수행한다.

## 핵심 특징

- **NumPy 기반 TF-IDF 직접 구현**: vocabulary 생성, document frequency, smoothed IDF, TF-IDF weighting과 L2 정규화를 직접 계산한다.
- **자체 희소 행렬 표현**: 전체 TF-IDF 행렬을 밀집 배열로 만들지 않고 `data`, `indices`, `indptr` 기반의 희소 표현으로 저장하고 연산한다.
- **코사인 유사도 기반 문서 검색**: L2 정규화된 문서와 질의 벡터의 희소 내적으로 관련 문서를 검색한다.
- **선형 SVM 문서 분류**: 분류 전용 TF-IDF 표현을 `SGDClassifier(loss="hinge")` 기반 다중 클래스 분류에 사용한다.
- **직접 구현에 대한 수치 검증**: 동일한 vocabulary와 설정을 사용하는 scikit-learn `TfidfVectorizer`와 비교해 수치적 일치 여부를 검증한다.
- **Privacy-conscious 데이터 처리**: headers, footers, quotes를 제거하고 structured identifier를 deterministic하게 redaction하며, 전체 정제 문서는 runtime artifact에 저장하지 않는다.
- **재현 가능한 실험 파이프라인**: 고정된 데이터 분할과 random seed로 분류 지표, TF-IDF 검증값, 희소 행렬·privacy 처리 통계를 기록한다.

## 아키텍처

프로젝트는 **오프라인 build 단계**와 **온라인 검색 단계**를 분리한다.

```mermaid
flowchart TD
    A["20 Newsgroups"] --> B["데이터 로딩<br/>headers / footers / quotes 제거"]
    B --> C["Privacy boundary<br/>structured-PII redaction"]
    C --> D["영문 전처리"]
    D --> E["분류: stratified 80/20 분할"]
    D --> F["검색: 전체 말뭉치 100% fit"]
    E --> G["분류 vocabulary + IDF fit<br/>(train 문서만)"]
    F --> H["검색 vocabulary + IDF fit<br/>(전체 문서)"]
    G --> I["분류 NumPy TF-IDF"]
    H --> J["검색 NumPy TF-IDF"]
    I --> K["분류 SparseMatrix"]
    J --> L["검색 SparseMatrix"]
    K --> M["scikit-learn TF-IDF와<br/>분류 공간 수치 검증"]
    L --> N["scikit-learn TF-IDF와<br/>검색 공간 수치 검증"]
    K --> O["Linear SVM 학습·평가"]
    M --> P["평가 보고서"]
    N --> P
    O --> P
    L --> Q["Runtime 검색 artifact"]
    Q --> R["사용자 Query"]
    R --> S["동일한 전처리 + 검색 TF-IDF transform"]
    S --> T["Sparse cosine similarity"]
    T --> U["Top-K 문서"]
```

분류 vocabulary와 IDF는 학습 문서에만 fit하고 테스트 문서를 그 공간으로 transform해 평가 단계의 data leakage를 막는다. 검색은 별도의 vocabulary와 IDF를 전체 정제 말뭉치에 fit해 100% 문서 검색 행렬을 만든다. 따라서 분류와 검색은 독립된 vocabulary/IDF 공간을 사용하며, 검색 시에는 build가 저장한 runtime artifact만 사용한다.

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

scikit-learn의 `fetch_20newsgroups(subset="all")`에서 제공하는 고정된 20개 카테고리 전체를 사용한다. 카테고리 제한을 개인정보 보호 수단으로 쓰지 않으며, 모든 카테고리에 같은 privacy pipeline을 적용한다.

- `alt.atheism`, `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`
- `misc.forsale`, `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`
- `sci.crypt`, `sci.electronics`, `sci.med`, `sci.space`, `soc.religion.christian`
- `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`, `talk.religion.misc`

`sci.med`도 이 계약에 포함된다. 현재 structured redaction은 이메일·전화번호·URL·IP 같은 형식화된 식별자만 대상으로 하므로, 자유 형식의 건강 정보는 제거하지 않는다.

### Privacy pipeline과 데이터 경계

loader는 headers, footers, quotes를 제거한 뒤 이메일, 명확한 전화번호, URL과 유효한 IPv4·IPv6를 deterministic하게 redaction한다. redaction과 영문 정규화 뒤 빈 문서만 제외한다. structured identifier만 제거하므로 PII-free를 보증하지 않으며, 특히 자유 형식의 건강 정보는 잔존할 수 있다.

정제된 전체 문서는 실행 중 TF-IDF fit/transform, 분류와 검색 행렬 생성에만 사용하고 저장하지 않는다. runtime metadata, 검색 결과와 오분류 보고서에는 structured-PII sanitization을 다시 적용하고 검증한 최대 240자의 snippet만 기록한다. 공개 검색 문서 ID는 filtering 전 loader 행 번호를 유지한다.

| 처리 | 근거 |
|---|---|
| headers, footers, quotes 제거 | 불필요한 metadata 노출과 metadata 과적합을 줄여 더 현실적인 분류 평가를 한다. ([scikit-learn 권장](https://scikit-learn.org/stable/datasets/real_world.html)) |
| 이메일·전화번호·URL·유효한 IP redaction | 직접 식별·연락에 쓰일 수 있는 structured identifier를 재현 가능한 규칙으로 제거한다. |
| 자유형 엔터티 미탐지 | 검증되지 않은 추측성 이름 규칙은 오탐과 어휘 왜곡을 낳을 수 있어 적용하지 않는다. |
| full text 비저장, 최대 240자 snippet | 원문 재배포와 불필요한 노출을 줄이면서 검색 결과를 확인할 수 있게 한다. |
| 빈 문서만 제외 | 주관적 문서 제거 대신 제외 수와 카테고리별 유지율을 build마다 기록한다. |

### 실제 privacy 처리 결과

| 범위 | raw | retained | excluded | 유지율 |
|---|---:|---:|---:|---:|
| 전체 | 18,846 | 18,309 | 537 | 97.15% |
| `sci.med` | 990 | 960 | 30 | 96.97% |

카테고리별 raw·retained·excluded 수는 `privacy_report.json`의 20개 항목에 기록한다. 이 수치는 structured redaction의 결과이지, 자유 형식 개인정보가 없다는 보장은 아니다.

| 제외 사유 | 문서 수 |
|---|---:|
| sanitization 후 빈 문서 | 537 |

| redaction 종류 | 횟수 |
|---|---:|
| email | 6,405 |
| phone | 2,157 |
| URL | 2 |
| IPv4 | 759 |
| IPv6 | 253 |

데이터 출처, 귀속, 라이선스 및 파생 정책은 [DATASET_LICENSE.md](DATASET_LICENSE.md)에 기록했다.

## 전처리와 TF-IDF

`EnglishPreprocessor`는 모든 문서와 검색 쿼리에 같은 순서를 적용한다.

1. 소문자로 변환한다.
2. 영문자와 공백 외 문자를 공백으로 치환한다.
3. whitespace로 분할한다.
4. 한 글자 토큰과 고정 불용어를 제거한다.

동의어 치환은 사용하지 않는다. 외부 어휘 자원에 새로운 의미 가정을 의존하게 되고, 문맥에 맞지 않는 치환이 생길 수 있으며, 직접 구현과 scikit-learn 검증의 분석 경계도 불필요하게 달라지기 때문이다.

불용어 적용·미적용의 이전 수치 비교는 세 카테고리 범위에서만 수행된 기록이므로 현재 20-category 결과로 일반화하지 않는다. 전체 카테고리 loader를 쓰는 방법과 현재 Python 검색 루프의 계산 한계는 [stop-word ablation](docs/stop-word-ablation.md)에 기록했다.

TF-IDF는 다음 정의를 NumPy 배열과 자체 `SparseMatrix`로 계산한다.

```text
TF(t, d)       = count(t, d)
IDF(t)         = log((1 + N) / (1 + df(t))) + 1
TF-IDF(t, d)   = TF(t, d) × IDF(t)
L2 정규화       = TF-IDF(d) / ||TF-IDF(d)||₂
```

분류는 학습 문서에서만 vocabulary와 IDF를 fit하고 테스트 문서를 그 공간으로 transform한다. 검색은 전체 정제 말뭉치에서 별도의 vocabulary와 IDF를 fit한다. 두 공간은 각각 scikit-learn `TfidfVectorizer`와 비교하며 최대 절대 오차 허용치는 `1e-6`이다. 검증 시 직접 구현의 raw-count TF, smoothed IDF, L2 정규화와 맞추기 위해 다음 설정을 고정한다.

```text
smooth_idf=True
sublinear_tf=False
norm="l2"
use_idf=True
dtype=float64
```

### TF-IDF 단계별 계산 예시와 행렬 매핑

`stage_example.json`은 분류 학습 행렬에서 첫 번째 비어 있지 않은 행을 골라 실제 중간값을 기록한다. 현재 예시는 train split 14,647개 문서로 IDF를 학습한 행 0이며, 아래 값은 별도로 다시 계산한 예시가 아니라 build에 사용된 배열에서 추출한 값이다.

| 단어 | vocabulary 열 | TF | IDF | 정규화 전 TF-IDF | L2 정규화 후 TF-IDF |
|---|---:|---:|---:|---:|---:|
| `acceleration` | 336 | 1.0 | 7.333963 | 7.333963 | 0.206474 |
| `analysis` | 2,454 | 1.0 | 5.548634 | 5.548634 | 0.156211 |
| `anything` | 2,955 | 1.0 | 3.672375 | 3.672375 | 0.103389 |
| `build` | 8,484 | 1.0 | 5.039100 | 5.039100 | 0.141866 |
| `buy` | 8,796 | 1.0 | 4.353734 | 4.353734 | 0.122571 |

예를 들어 `acceleration`은 해당 문서에 한 번 등장하므로 TF는 1이다. 전체 학습 문서의 document frequency로 계산한 IDF 7.333963을 곱하면 정규화 전 TF-IDF도 7.333963이며, 문서 벡터 전체를 L2 norm으로 나누면 최종 가중치는 0.206474가 된다.

TF-IDF 행렬의 shape은 `문서 수 × vocabulary 크기`다. 행은 split 내부 문서 위치, 열은 정렬된 `feature_names_`의 단어 위치이며, `(행 0, 열 336)`은 위 문서의 `acceleration` 가중치를 뜻한다. 분류 train/test 행렬은 각각 `14,647 × 78,613`, `3,662 × 78,613`이고, 검색 행렬은 `18,309 × 91,538`이다.

테스트 문서는 train vocabulary와 IDF로만 transform하고, 검색 쿼리는 full-corpus 검색 vocabulary와 IDF로 transform한다. 따라서 문서와 쿼리의 같은 열이 항상 같은 단어를 나타내며, vocabulary에 없는 쿼리 단어는 무시된다. 독립적으로 fit한 분류 벡터와 검색 벡터는 열 의미와 IDF가 다르므로 서로 직접 비교하지 않는다.

## 검색과 분류

문서와 쿼리 벡터는 L2 정규화되어 있으므로 코사인 유사도는 내적과 같다. `sparse_dot`은 `np.intersect1d(..., return_indices=True)`로 공통 열의 위치를 찾고 `np.dot`으로 대응 TF-IDF 값을 계산한다. 동점은 공개 source document ID 오름차순으로 정렬한다.

분류는 `SGDClassifier(loss="hinge")`를 사용한다. 프로젝트와 테스트 코드는 SciPy를 직접 import하지 않으며, 학습과 예측 때 자체 희소 행렬의 행을 `batch_size` 이하 NumPy 밀집 배열로만 변환한다. 전체 말뭉치를 한 번에 밀집화하지 않는다.

## 재현 결과

고정 설정은 20개 카테고리 전체의 stratified 8:2 분할과 `random_state=42`다. 현재 저장된 보고서는 다음 결과를 기록한다.

| 항목 | 재빌드 결과 |
|---|---:|
| 정제 후 문서 | 18,309 |
| 학습 / 테스트 문서 | 14,647 / 3,662 |
| 분류 vocabulary | 78,613 |
| 검색 vocabulary | 91,538 |
| 검색 전체 행렬 shape | 18,309 × 91,538 |
| `nnz` | 1,230,094 |
| 밀도 / 희소율 | 0.073395977% / 99.926604023% |
| 밀집 / 자체 희소 저장 크기 | 13,407,753,936 / 14,834,368 bytes |
| 밀집 대비 저장 효율 | 903.8304790605167배 |
| 분류 TF-IDF 최대 절대 오차 | 5.440092820663267e-15 |
| 검색 TF-IDF 최대 절대 오차 | 9.2148511043888e-15 |
| Accuracy | 0.7670671764063354 |
| macro F1 | 0.7543850883245388 |
| 전체 오분류 | 853 |

`space shuttle orbit`의 Top-5도 모두 `sci.space`였다.

| 순위 | 점수 | source document ID | 카테고리 |
|---:|---:|---:|---|
| 1 | 0.413791 | 4,672 | `sci.space` |
| 2 | 0.384463 | 4,389 | `sci.space` |
| 3 | 0.373090 | 5,788 | `sci.space` |
| 4 | 0.368089 | 12,888 | `sci.space` |
| 5 | 0.364911 | 10,571 | `sci.space` |

![20개 카테고리 혼동 행렬](artifacts/reports/confusion_matrix.png)

혼동 행렬은 20 × 20이며, 행과 열은 위의 고정된 20개 카테고리 순서를 사용한다.

### `talk.*.misc` 클래스 오분류 분석

`talk.politics.misc`와 `talk.religion.misc`는 현재 분류 결과에서 recall이 특히 낮다. 두 클래스 모두 예측된 문서는 비교적 정확하지만 실제 문서 상당수가 인접 주제 클래스로 빠지는 **과소 예측** 양상을 보인다.

| 실제 클래스 | 테스트 문서 | Precision | Recall | F1 | 주요 오분류 |
|---|---:|---:|---:|---:|---|
| `talk.politics.misc` | 151 | 0.8061 | 0.5232 | 0.6345 | `talk.politics.mideast` 24건, `talk.politics.guns` 19건 |
| `talk.religion.misc` | 121 | 0.6852 | 0.3058 | 0.4229 | `soc.religion.christian` 27건, `alt.atheism` 14건 |

주된 원인은 다음과 같다.

- `misc`는 하나의 좁은 주제라기보다 각 계열의 잔여 주제를 모은 범주이므로 고유 어휘 경계가 약하고 인접 클래스와 의미가 겹친다.
- 정제 후 문서 수가 `talk.politics.misc` 755건, `talk.religion.misc` 605건으로, 각각의 인접 클래스보다 적다.
- headers, footers, quotes 제거는 metadata 과적합과 정보 노출을 줄이지만 newsgroup을 직접 식별할 수 있는 단서도 함께 제거한다.
- 현재 unigram TF-IDF는 구문과 문맥을 표현하지 못하며, 6 epoch mini-batch SGD 학습은 두 잔여 클래스를 충분히 분리하지 못한다.

동일한 전처리·분할에서 진단용 `LinearSVC`를 사용하면 전체 Accuracy는 0.7671에서 0.7851로, 두 클래스의 recall은 각각 0.6490과 0.5041로 상승했다. 따라서 낮은 성능은 TF-IDF 계산 오류가 아니라 클래스 자체의 의미 중첩과 표본 수 차이가 주원인이며, 현재 학습 방식이 이를 추가로 악화한 결과로 해석한다. 직접 구현한 TF-IDF가 scikit-learn과 `1e-6` 이내로 일치한다는 수치 검증과 분류 성능은 서로 다른 검증 대상이다.

### 오분류 사례 5건 분석

아래 분석은 `misclassifications.json`의 정제된 최대 240자 스니펫을 기준으로 한다. 스니펫은 원문의 전체 문맥을 보존하지 않으므로, 관찰 가능한 단서만으로 원인을 해석했다.

| source document ID | 실제 → 예측 | 관찰한 실패 원인 | BoW/TF-IDF 한계와 개선 방향 |
|---:|---|---|---|
| 10,167 | `alt.atheism` → `rec.autos` | `car drivers`, `non car owner`처럼 자동차 단어만 남고 무신론 주제 단서가 없다. | 주제 이탈이나 제거된 문맥은 단어 빈도로 복원할 수 없다. 정보량이 낮은 문서는 신뢰도를 표시해 별도 검토한다. |
| 17,783 | `sci.med` → `rec.sport.baseball` | `local BBB`, `business bureau`라는 짧고 일반적인 신고 문장뿐이며 의학 단어가 없다. | 희소한 일반 단어만 있는 문서는 클래스 경계가 불안정하다. 최소 유효 토큰 수와 예측 margin을 함께 평가한다. |
| 2,608 | `comp.os.ms-windows.misc` → `comp.graphics` | `FAQ`, `telnet`, `FTP`, `internet` 등 일반 컴퓨터 용어가 중심이고 Windows 고유 단서가 약하다. | 인접 기술 클래스가 vocabulary를 공유한다. 단어 bi-gram과 문자 n-gram을 비교해 복합 제품명·표현을 보존한다. |
| 3,292 | `talk.politics.misc` → `talk.politics.guns` | `BATF`, `assault`, `court`, `search`가 총기 정책 문서에서 강하게 나타날 법한 단서다. | `misc`의 넓은 라벨과 guns 하위 주제가 의미상 겹친다. 클래스별 recall과 confusion pair를 기준으로 모델과 라벨 경계를 점검한다. |
| 15,723 | `talk.religion.misc` → `alt.atheism` | 관찰과 지각에 관한 철학적 표현이 중심이고 특정 종교를 가리키는 단어가 없다. | BoW는 주장 대상과 화자의 관점을 구분하지 못한다. 문장 문맥을 반영하는 임베딩 모델과 동일 split에서 비교한다. |

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
| `artifacts/reports/confusion_matrix.png` | 20 × 20 혼동 행렬 |

## 한계와 개선 방향

### BoW, 다의어와 문맥 임베딩

BoW/TF-IDF는 같은 vocabulary 열의 출현 횟수를 사용하므로 어순과 주변 문맥을 직접 표현하지 않는다. 예를 들어 `dog bites man`과 `man bites dog`는 의미가 다르지만 동일한 세 단어가 한 번씩 등장해 같은 BoW가 된다. `the bank approved the loan`과 `we sat on the river bank`의 `bank`도 금융기관과 강둑이라는 서로 다른 의미가 같은 열에 누적된다.

이 한계는 검색에서 표면 단어만 같은 문서를 상위에 놓고, 분류에서는 문서의 대상·관점보다 자주 등장한 주제 단어에 의존하게 만든다. 단어 bi-gram은 일부 어순과 구를 보존하지만 멀리 떨어진 문맥과 새로운 표현까지 해결하지는 못한다.

문맥 임베딩은 주변 단어와 어순을 함께 사용해 문장 또는 문서 벡터를 만들기 때문에 같은 `bank`도 문맥에 따라 다른 표현을 만들 수 있다. 후속 실험에서는 현재 unigram TF-IDF + SGD를 기준선으로 두고, 단어 bi-gram 선형 모델과 문맥 임베딩 기반 분류·검색을 동일한 privacy 처리, train/test split과 query set에서 비교한다. 임베딩 모델을 도입하더라도 라벨 중첩과 주제 이탈 문서를 자동으로 해결한다고 가정하지 않고 오분류를 다시 점검한다.

### 100만 건 확장성과 정량 개선 계획

현재 검색 행렬은 문서 18,309개에 `nnz` 1,230,094개로 문서당 평균 약 67.2개의 nonzero를 가진다. 같은 평균과 현재 vocabulary 91,538개가 유지된다고 가정하면 100만 문서에서는 약 6,718만 `nnz`가 필요하다. 현재 `float64 data + int32 indices + indptr` 구조의 단순 추정치는 약 810 MB지만, 같은 행렬을 밀집 표현으로 만들면 약 732 GB가 필요하다. 실제 vocabulary 증가와 Python 객체·metadata 비용은 이 추정에 포함되지 않는다.

더 큰 병목은 검색 시 모든 문서를 Python loop로 순회하는 구조다. 100만 문서는 현재보다 약 54.6배 많은 행을 검사하므로, 다른 조건이 같아도 검색 시간이 같은 비율로 증가할 가능성이 있다. 현재는 latency benchmark가 없으므로 이 비율은 실측 응답시간이 아니라 선형 복잡도에 따른 추정이다.

확장 단계에서는 다음 순서로 비교한다.

1. 희소 TF-IDF는 query term이 등장한 문서만 방문하도록 inverted index를 만들고 Top-K heap으로 전체 score 정렬을 피한다.
2. 문서 ID 범위로 shard한 뒤 shard별 Top-K를 병합해 메모리와 검색 부하를 분산한다.
3. 문맥 임베딩을 채택할 경우 ANN index를 사용하고 정확 검색 대비 Recall@K 손실과 latency 이득을 함께 측정한다.
4. build는 chunk 단위로 처리하고 peak RSS, index 크기와 build 시간을 기록해 전체 행렬의 동시 메모리 상주를 피한다.

비교 실험은 현재 Accuracy 0.7671과 macro F1 0.7544를 분류 기준선으로 고정한다. 같은 split과 seed에서 Accuracy, macro F1, 클래스별 recall을 비교하고, 고정된 held-out query set에서 Precision@10과 MAP@10을 측정한다. 검색 규모를 18,309개, 10만 개, 100만 개로 늘리며 build 시간, index byte, peak RSS, query latency p50/p95를 기록한다. ANN은 추가로 Recall@10을 보고해 속도 향상이 검색 품질 손실을 숨기지 않게 한다. 현재 Top-5의 카테고리 일치는 정성적 예시이므로 이 정량 평가의 대체물이 아니다.

structured identifier regex는 자유 형식의 사람 이름·주소·건강정보를 탐지하지 않는다. 이 pipeline은 PII-free 보증이 아니며, 실제 배포에는 별도의 데이터 거버넌스 검토가 필요하다.
