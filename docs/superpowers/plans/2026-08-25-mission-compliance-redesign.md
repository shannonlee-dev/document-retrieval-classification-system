# Mission Compliance Redesign Implementation Plan

**Goal:** 미션의 NumPy TF-IDF·검색·분류 요구사항을 유지하면서 ML vocabulary와 개인정보 처리 책임을 분리한다.

**Architecture:** 선택한 20 Newsgroups 문서에서 metadata를 제거하고 deterministic structured PII를 redaction한다. 정제된 전체 문서는 메모리 내 ML 입력에만 사용하고, artifact와 report에는 재정제한 최대 240자 snippet만 저장한다.

**Scope:** `comp.graphics`, `rec.sport.baseball`, `sci.space`; email, 명확한 phone, URL, 유효한 IPv4·IPv6; privacy 처리 통계. 자유형 사람 이름·주소 NER, BM25, 역색인과 보너스 TF 변형은 제외한다.

## Constraints

- `SAFE_TERMS`나 다른 고정 주제어 allowlist를 두지 않는다.
- headers, footers, quotes는 scikit-learn loader에서 제거한다.
- 원문과 정제된 전체 문서를 runtime/report artifact에 저장하지 않는다.
- snippet은 최대 240자이며 저장 직전에 structured-PII 검사를 다시 통과해야 한다.
- filtering 후 `source_doc_id`는 filtering 전 loader row를 계속 가리킨다.
- NumPy TF-IDF, 자체 `SparseMatrix`, sklearn 수치 검증, cosine 검색, bounded NumPy batch SVM과 기존 CLI를 유지한다.
- privacy pipeline을 PII-free 보증으로 표현하지 않는다.

## Task 1: Structured privacy boundary

**Files:** `src/document_system/privacy.py`, `src/document_system/dataset.py`, `tests/test_privacy.py`, `tests/test_dataset.py`

- [x] 일반 비민감 어휘가 삭제되지 않는 실패 테스트를 추가한다.
- [x] email, phone, URL, IPv4와 IPv6 redaction 및 종류별 count를 테스트한다.
- [x] 자유형 이름·주소·건강정보를 추측해 문서를 제외하지 않는 정책을 테스트한다.
- [x] sanitization 후 빈 문서를 제외하고 stable source ID를 유지한다.
- [x] raw/retained/dropped, 유지율, redaction, 카테고리별 count를 집계한다.

## Task 2: Full text와 snippet 경계

**Files:** `src/document_system/pipeline.py`, `src/document_system/artifacts.py`, `src/document_system/search.py`, `src/document_system/classification.py`

- [x] 정제된 전체 문서로 TF-IDF fit/transform, 분류와 검색 행렬을 생성한다.
- [x] 전체 문서에서 최대 240자의 sanitized snippet을 별도로 만든다.
- [x] runtime metadata, 검색 예시와 오분류 보고서에는 snippet만 전달한다.
- [x] artifact save/load와 `DocumentSearch`에서 snippet 길이와 privacy policy를 검증한다.
- [x] 이전 `safe-topic-terms-v1` artifact를 재빌드 안내와 함께 거부한다.

## Task 3: Privacy evidence와 문서

**Files:** `artifacts/reports/dataset_sanitization_report.json`, `README.md`, `DATASET_LICENSE.md`

- [x] build에서 실제 측정한 privacy report를 생성한다.
- [x] 세 카테고리 선택 이유와 structured redaction 정책을 설명한다.
- [x] full text 비저장과 bounded snippet 경계를 설명한다.
- [x] 자유형 엔터티 탐지 미지원과 PII-free 비보증을 한계로 기록한다.
- [x] 실제 build의 문서·행렬·분류·검증 수치로 README를 갱신한다.

## Task 4: Verification

- [x] `PYTHONPATH=src .venv/bin/pytest -q`
- [x] `.venv/bin/ruff check src tests`
- [x] 실제 dataset build와 `dataset_sanitization_report.json` 생성
- [x] TF-IDF 최대 오차 `1e-6` 이하 확인
- [x] runtime metadata에 full document 필드가 없고 snippet이 240자 이하인지 확인
- [x] CLI 검색과 분류 report 생성 확인

## Acceptance Criteria

- raw 2,954개 중 최소 500개와 세 카테고리를 유지한다.
- vocabulary가 고정 68-term allowlist에 의존하지 않는다.
- privacy report count는 실제 loader 처리 결과와 일치한다.
- 카테고리별 sanitization 영향이 한 클래스를 붕괴시키지 않는다.
- 저장되는 텍스트는 bounded sanitized snippet뿐이다.
- 전체 테스트와 Ruff가 통과하고 실제 build가 재현된다.
