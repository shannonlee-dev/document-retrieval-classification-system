# Mission Compliance Redesign

## Goal

보너스 항목을 제외한 `docs/private/mission.md` 요구사항에 맞게 데이터 정제, 검색, 분류, 산출물과 문서를 개편한다. 기존 CLI 사용법은 유지하되 개인정보가 포함될 수 있는 원문은 모델 입력과 산출물에 남기지 않는다.

## Scope

다음 여덟 항목을 모두 해결한다.

1. 코사인 유사도를 NumPy 벡터 연산으로 계산한다.
2. 프로젝트 코드와 테스트의 직접적인 SciPy 사용을 제거한다.
3. 모델에 전달되는 데이터에서 식별 가능한 이름, 연락처, 건강정보를 배제한다.
4. 런타임 산출물에 전체 원문을 저장하지 않는다.
5. 빈 문서를 데이터셋에서 제거한다.
6. 데이터 출처, 라이선스, 저작권 처리 근거를 문서화한다.
7. 동의어 치환을 하지 않는 기준과 이유를 README에 설명한다.
8. 영어 토큰화를 정제 후 공백 분할 방식으로 변경한다.

보너스 TF 변형, BM25, 역색인은 구현하지 않는다.

## Compatibility Decision

- `python main.py build`와 `python main.py --query "..." --topk 5` CLI는 유지한다.
- `NumpyTfidfVectorizer`의 공개 메서드와 `SearchResult` 출력 필드는 유지한다.
- 데이터 모델과 검색 artifact 내부 구조는 변경한다.
- `ARTIFACT_VERSION`은 현재 값 `1`을 유지한다.
- `build`는 기존 `artifacts/runtime/matrix.npz`, `artifacts/runtime/metadata.json`과 `artifacts/reports/*`를 새 형식과 결과로 덮어쓴다.
- 변경 전 runtime artifact의 호환 로딩은 지원하지 않는다. 새 필수 필드가 없으면 명확한 재빌드 오류를 반환한다.

## Dataset and Privacy Boundary

### Source and categories

Scikit-learn의 20 Newsgroups loader를 사용하되 미션 예시에 포함된 다음 세 카테고리만 선택한다.

- `comp.graphics`
- `rec.sport.baseball`
- `sci.space`

이 구성은 최소 두 카테고리와 500개 문서 조건을 만족하면서 건강정보 중심 카테고리인 `sci.med`를 사용하지 않는다.

### Sanitization

`privacy.py`는 ML vocabulary와 독립된 structured-PII 경계를 소유한다. loader에서 받은 원문은 다음 순서로 즉시 변환한다.

1. email, 명확한 phone, URL과 유효한 IPv4·IPv6를 deterministic하게 redaction한다.
2. 영문 소문자 단어를 공백으로 연결해 정제된 전체 문서를 만든다.
3. 결과가 빈 문서는 제거한다.
4. 전체 문서는 메모리 내 TF-IDF, 분류와 검색 행렬 생성에만 사용한다.
5. artifact와 report에는 같은 sanitization을 다시 통과한 최대 240자의 snippet만 저장한다.

사람 이름·주소 같은 자유형 엔터티 detector와 의심 문서 전체 삭제는 범위에 포함하지 않는다. 이 설계는 PII-free를 보증하지 않으며, 카테고리 선택, metadata 제거, structured redaction, full-text 비저장과 bounded snippet으로 잔여 노출 범위를 줄인다.

### Dataset model

`DatasetBundle`은 다음 값을 보관한다.

- `texts`: 저장하지 않는 메모리 내 ML 입력용 정제 전체 문서
- `labels`: 연속된 정수 라벨
- `target_names`: 세 카테고리명
- `source_doc_ids`: loader 결과에서의 원본 위치를 나타내는 정수 ID

`validate_dataset`은 공백뿐인 문서를 거부하고 문서, 라벨, 원본 ID의 길이 일치를 검증한다.

## Preprocessing and TF-IDF

`EnglishPreprocessor.tokenize`는 다음 과정을 사용한다.

1. 소문자 변환
2. 영문자와 공백 이외 문자를 공백으로 치환
3. `split()`을 이용한 공백 토큰화
4. 길이 1 이하 토큰과 불용어 제거

TF, IDF, TF-IDF, L2 정규화는 현재 NumPy 기반 구현과 공개 인터페이스를 유지한다. Scikit-learn `TfidfVectorizer`는 기존 검증 모듈에서만 참조 구현으로 사용한다.

## Search

`sparse_dot`은 정렬된 희소 열 번호의 교집합을 `np.intersect1d(..., return_indices=True)`로 구하고 대응 값을 `np.dot`으로 계산한다. 문서와 쿼리가 이미 L2 정규화되어 있으므로 이 내적을 코사인 점수로 사용한다.

`DocumentSearch`는 전체 원문 대신 다음을 받는다.

- 비식별 `snippets`
- 공개 검색 ID로 사용할 `document_ids`

검색 결과의 `score`, `doc_id`, `label`, `text_snippet` 형식과 Top-k 정렬 규칙은 유지한다.

## Classification Without Direct SciPy

프로젝트에서 `scipy.sparse`를 직접 import하지 않는다. 자체 `SparseMatrix`에 선택 행을 NumPy 밀집 배열로 변환하는 기능을 사용하고, 다음 단위로만 밀집화한다.

- 학습: `batch_size`개의 행
- 예측: `batch_size`개의 행

`SGDClassifier(loss="hinge")`, 8:2 계층 분할, 고정 `random_state`, Accuracy, macro F1과 혼동 행렬은 유지한다. 전체 TF-IDF 행렬을 한 번에 밀집화하지 않는다.

## Artifacts

`metadata.json`에는 원문이나 정제된 전체 `texts`를 저장하지 않는다. 다음 값만 기록한다.

- shape와 dtype
- feature names와 stop words
- 재정제된 최대 240자의 `snippets`
- `document_ids`
- labels와 target names
- privacy policy identifier

loader는 이 필수 필드를 검증한다. 변경 전 metadata를 발견하면 `python main.py build`를 다시 실행하라는 오류를 낸다. 버전 숫자는 올리지 않고 정상 build가 기존 파일을 덮어쓴다.

평가 보고서의 검색 예시와 오분류 예시도 같은 제한 snippet만 포함한다. build는 실제 문서 입력·유지·빈 문서 제외 수, structured redaction 수와 카테고리별 처리 수를 `privacy_report.json`에 기록한다.

## Documentation and Licensing

README는 새 문서 수, 세 카테고리, structured redaction, full-text/snippet 저장 경계, 빈 문서 제거, NumPy 코사인 연산, NumPy 분류 배치와 변경된 실험 결과를 설명한다.

동의어 치환은 외부 사전 의존성, 문맥에 따른 잘못된 치환, sklearn 검증과의 분석 경계 복잡화를 피하기 위해 적용하지 않는다고 명시한다.

`DATASET_LICENSE.md`를 추가해 다음을 기록한다.

- Twenty Newsgroups 출처와 UCI DOI
- UCI가 표시한 CC BY 4.0 조건과 귀속
- 원문을 저장소에 재배포하지 않는 정책
- 정제된 전체 문서는 저장하지 않고 ML 입력으로만 사용하는 방식
- 보고서와 runtime artifact에는 제한 길이 snippet만 저장한다는 정책

## Error Handling

- 정제 후 문서가 500개 미만이거나 카테고리가 두 개 미만이면 build를 중단한다.
- 공백 문서와 길이가 맞지 않는 라벨·문서 ID는 검증 오류로 처리한다.
- sanitization 후 빈 문서는 dataset에서 제외한다.
- 변경 전 runtime artifact는 재빌드 안내와 함께 로드를 거부한다.
- 쿼리에 학습 vocabulary 단어가 없으면 기존과 같이 사용자 오류를 반환한다.

## Testing

변경은 테스트 우선으로 진행한다.

1. 공백 문서 거부와 원본 ID 길이 검증 테스트
2. 일반 어휘 보존, structured PII redaction과 빈 결과 제거 테스트
3. 데이터 loader가 세 카테고리, 500개 이상, 비어 있지 않은 정제 문서와 정확한 privacy 통계를 반환하는 테스트
4. `sparse_dot`이 `np.dot` 기반 기대값과 같은 결과를 내는 테스트
5. 프로젝트와 테스트에 직접적인 SciPy import가 없고 NumPy 배치로 분류되는 테스트
6. 전체 정제 문서는 ML vocabulary에 반영되지만 artifact에는 없고 제한 snippet과 원본 행 ID만 저장·복원하는 테스트
7. CLI 검색 출력과 소규모 전체 pipeline 통합 테스트
8. 전체 테스트 실행
9. 전체 dataset build 후 보고서 수치, 개인정보 패턴, 빈 문서와 산출물 형식 점검

## Acceptance Criteria

- 여덟 개선 항목이 코드, 테스트, README와 산출물에 반영된다.
- 보너스 기능은 추가하지 않는다.
- CLI 사용법은 바뀌지 않는다.
- runtime과 report 파일은 기존 경로에 덮어쓴다.
- 코드와 테스트에 직접적인 `scipy` import가 없다.
- 저장된 snippet은 비어 있지 않고 240자 이하이며 저장 직전 structured-PII 검사를 통과한다.
- vocabulary는 고정 주제어 allowlist에 의존하지 않고 일반 어휘를 보존한다.
- TF-IDF 최대 오차가 `1e-6` 이내다.
- 검색 Top-5와 분류 평가 산출물이 생성된다.
- 전체 테스트가 통과한다.
