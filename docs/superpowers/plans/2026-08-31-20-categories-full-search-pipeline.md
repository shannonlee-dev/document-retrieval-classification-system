# 20-Category Full-Corpus Search Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 20 Newsgroups 전체 20개 카테고리를 고정 사용하면서 분류는 누수 없는 80% fit을 유지하고 검색은 전체 문서 100%에 별도로 fit한 TF-IDF 행렬을 사용하게 만든다.

**Architecture:** `pipeline.py`는 build orchestration과 report persistence만 담당하고, 새 `build_stages.py`가 독립된 classification/search stage를 소유한다. Classification stage는 80/20 feature space에서 평가 결과만 반환해 대형 행렬 수명을 끝내고, search stage는 별도 vectorizer를 전체 corpus에 fit해 v2 runtime artifact를 만든다.

**Tech Stack:** Python 3.10+, NumPy, scikit-learn, Matplotlib, pytest 8, Ruff

**Spec:** `docs/superpowers/specs/2026-08-31-20-categories-full-search-pipeline-design.md`

## Global Constraints

- 실제 dataset build는 선택 옵션 없이 정확히 20개 카테고리를 요구한다.
- `build_from_dataset()`은 테스트와 재사용을 위해 두 개 이상의 카테고리를 계속 허용한다.
- Classification TF-IDF는 stratified train 80%에만 fit하고 test 20%는 transform만 한다.
- Search TF-IDF는 sanitization 후 유지된 전체 문서 100%에 독립적으로 fit한다.
- 두 vectorizer는 객체, vocabulary와 IDF를 공유하지 않는다.
- Classification과 search TF-IDF 모두 scikit-learn 대비 최대 절대 오차 `1e-6` 이하여야 한다.
- Runtime artifact는 v2만 로드하며 `fit_scope="full_corpus"`를 검증한다.
- headers/footers/quotes 제거, structured-PII redaction, full-text 비저장과 240자 snippet 정책은 변경하지 않는다.
- `sci.med`의 자유형 건강정보 잔여 위험은 수용하며 새 detector나 category별 억제를 추가하지 않는다.
- 기존 build/search CLI 인자 형식, NumPy `SparseMatrix`, cosine search와 SGD hinge-loss SVM을 유지한다.
- Stop-word ablation 알고리즘 최적화는 범위 밖이다. 현재 3-category 결과 파일을 20-category 현재 결과처럼 남기지는 않는다.
- 새 외부 의존성을 추가하지 않는다.
- 작업 시작 기준선은 `.venv/bin/pytest -q`의 `65 passed`와 `.venv/bin/ruff check src tests` 통과다.

## File Map

- Create: `src/document_system/build_stages.py` — classification/search build stage와 stage result 타입
- Modify: `src/document_system/dataset.py` — 전체 20-category loader와 실제 dataset 계약
- Modify: `src/document_system/pipeline.py` — stage orchestration, report persistence와 `BuildReport`
- Modify: `src/document_system/artifacts.py` — v2 full-corpus artifact metadata 및 검증
- Modify: `src/document_system/cli.py` — 두 vocabulary와 두 validation error 출력
- Modify: `src/document_system/ablation.py` — 전체 20-category dataset 설명
- Modify: `tests/test_dataset.py` — 전체 category 요청과 계약 테스트
- Create: `tests/test_build_stages.py` — 두 TF-IDF feature space의 독립성 테스트
- Modify: `tests/test_pipeline.py` — report, 전체 검색 행렬과 held-out token 통합 테스트
- Modify: `tests/test_artifacts.py` — artifact v2 round-trip 및 metadata 오류 테스트
- Modify: `tests/test_cli.py` — dual-space build 완료 출력 테스트
- Modify: `tests/test_ablation.py` — 전체 dataset 설명 회귀 테스트
- Modify: `README.md` — architecture, 실행 의미, 20-category 실제 결과와 한계
- Modify: `DATASET_LICENSE.md` — 전체 dataset 범위와 privacy 위험
- Modify: `docs/stop-word-ablation.md` — 20-category 방법과 성능 한계
- Delete: `artifacts/reports/stop_word_ablation.json` — 재실행하지 않은 3-category 결과 제거
- Regenerate: `artifacts/reports/{confusion_matrix.png,matrix_stats.json,metrics.json,misclassifications.json,privacy_report.json,search_examples.json,stage_example.json,tfidf_validation.json}`
- Runtime-only: `artifacts/runtime/{matrix.npz,metadata.json}` — v2 smoke test에 사용하며 `.gitignore` 상태 유지

---

### Task 1: Enforce the fixed 20-category dataset contract

**Files:**
- Modify: `src/document_system/dataset.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: existing `DatasetBundle`, `PrivacyReport`, `fetch_20newsgroups`
- Produces: `EXPECTED_CATEGORY_COUNT: int = 20`
- Produces: `validate_full_20_newsgroups(bundle: DatasetBundle) -> None`
- Preserves: `load_20newsgroups() -> DatasetBundle`, `_validate_dataset(...) -> None`

- [ ] **Step 1: Add failing tests for an unfiltered loader and the exact category contract**

In `tests/test_dataset.py`, capture the sklearn loader arguments and construct a 500-document, 20-label fixture:

```python
def make_full_category_fake() -> SimpleNamespace:
    target_names = [f"category.{index}" for index in range(20)]
    labels = np.repeat(np.arange(20, dtype=np.int32), 25)
    texts = [f"category token document {index}" for index in range(labels.size)]
    return SimpleNamespace(data=texts, target=labels, target_names=target_names)


def test_loader_requests_all_20_categories(monkeypatch) -> None:
    captured_options = None

    def fake_fetch(**options):
        nonlocal captured_options
        captured_options = options
        return make_full_category_fake()

    monkeypatch.setattr(dataset_module, "fetch_20newsgroups", fake_fetch)

    bundle = dataset_module.load_20newsgroups()

    assert captured_options is not None
    assert "categories" not in captured_options
    assert captured_options["subset"] == "all"
    assert len(bundle.target_names) == 20
    assert set(bundle.labels.tolist()) == set(range(20))
```

Add direct contract failures:

```python
def test_full_dataset_contract_rejects_missing_category(full_bundle) -> None:
    invalid = dataclasses.replace(
        full_bundle,
        labels=np.where(full_bundle.labels == 19, 18, full_bundle.labels),
    )

    with pytest.raises(ValueError, match="all 20 categories"):
        dataset_module.validate_full_20_newsgroups(invalid)


def test_full_dataset_contract_rejects_out_of_range_label(full_bundle) -> None:
    labels = full_bundle.labels.copy()
    labels[0] = 20

    with pytest.raises(ValueError, match="class IDs"):
        dataset_module.validate_full_20_newsgroups(
            dataclasses.replace(full_bundle, labels=labels)
        )
```

Update the existing loader privacy fixture to expose 20 target names and at least 25 retained documents per class. Keep the existing email/phone/drop assertions for the first three classes and assert that `privacy_report.category_counts` has exactly the 20 target-name keys.

- [ ] **Step 2: Run the dataset tests and verify the new tests fail**

Run: `.venv/bin/pytest tests/test_dataset.py -q`

Expected: FAIL because the loader still passes `categories=SAFE_CATEGORIES` and `validate_full_20_newsgroups` does not exist.

- [ ] **Step 3: Implement the exact full-dataset validator and remove the category filter**

In `src/document_system/dataset.py`, replace `SAFE_CATEGORIES` with the contract constant and validator:

```python
EXPECTED_CATEGORY_COUNT = 20


def validate_full_20_newsgroups(bundle: DatasetBundle) -> None:
    expected_class_ids = set(range(EXPECTED_CATEGORY_COUNT))
    observed_class_ids = set(np.asarray(bundle.labels, dtype=np.int64).tolist())
    privacy_categories = set(bundle.privacy_report.category_counts)
    target_categories = set(bundle.target_names)
    if len(bundle.target_names) != EXPECTED_CATEGORY_COUNT:
        raise ValueError("the full build requires exactly 20 target categories")
    if observed_class_ids != expected_class_ids:
        raise ValueError("the full build requires all 20 categories and class IDs 0..19")
    if privacy_categories != target_categories:
        raise ValueError("privacy report categories must match all target categories")
```

Call sklearn without a category selection:

```python
dataset = fetch_20newsgroups(
    subset=DATASET_SUBSET,
    remove=METADATA_FIELDS_TO_REMOVE,
    shuffle=True,
    random_state=DEFAULT_RANDOM_STATE,
)
```

Construct `bundle`, call `validate_full_20_newsgroups(bundle)`, then return it. In `build_project()`, call the same validator immediately after `load_20newsgroups()` so a monkeypatched loader cannot bypass the production contract. Do not add this call to `build_from_dataset()`.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/pytest tests/test_dataset.py tests/test_pipeline.py -q`

Expected: PASS, including the unchanged custom two-category `build_from_dataset()` seam.

- [ ] **Step 5: Run Ruff and commit the dataset contract**

Run: `.venv/bin/ruff check src/document_system/dataset.py tests/test_dataset.py tests/test_pipeline.py`

Expected: `All checks passed!`

Commit:

```bash
git add src/document_system/dataset.py tests/test_dataset.py tests/test_pipeline.py
git -c user.name=shannonlee-dev commit -m "feat(dataset): load all 20 newsgroups categories"
```

---

### Task 2: Version and validate full-corpus search artifacts

**Files:**
- Modify: `src/document_system/artifacts.py`
- Modify: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: existing `save_search_artifacts(...)`, `load_search_artifacts(...)`
- Produces: `ARTIFACT_VERSION = 2`, `SEARCH_FIT_SCOPE = "full_corpus"`
- Preserves: `SearchArtifacts` fields and artifact filenames
- Produces metadata keys: `fit_scope`, `fit_document_count`, `category_count`

- [ ] **Step 1: Write failing v2 metadata and validation tests**

Extend the round-trip assertion in `tests/test_artifacts.py`:

```python
metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
assert metadata["artifact_version"] == 2
assert metadata["fit_scope"] == "full_corpus"
assert metadata["fit_document_count"] == matrix.shape[0]
assert metadata["category_count"] == len(target_names)
```

Add metadata mutation tests:

```python
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_version", 1),
        ("fit_scope", "train_split"),
        ("fit_document_count", 999),
        ("category_count", 999),
    ],
)
def test_search_artifacts_reject_incompatible_build_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    vectorizer, matrix, snippets, labels, target_names, document_ids = make_search_data()
    save_search_artifacts(
        tmp_path, vectorizer, matrix, snippets, labels, target_names, document_ids
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="rebuild"):
        load_search_artifacts(tmp_path)
```

Add a label-range test by changing one stored label to `len(target_names)`, and add a save-time row-count test using one fewer snippet than matrix rows.

- [ ] **Step 2: Run artifact tests and verify failure**

Run: `.venv/bin/pytest tests/test_artifacts.py -q`

Expected: FAIL because v1 metadata lacks the new full-corpus fields and validations.

- [ ] **Step 3: Implement artifact v2 metadata and invariants**

In `src/document_system/artifacts.py`:

```python
ARTIFACT_VERSION = 2
SEARCH_FIT_SCOPE = "full_corpus"
REQUIRED_METADATA_FIELDS = frozenset(
    {
        "snippets",
        "document_ids",
        "privacy_policy",
        "fit_scope",
        "fit_document_count",
        "category_count",
    }
)
```

Before writing, convert labels/document IDs once, verify all row-aligned inputs, require non-empty target names, and reject labels outside `0..len(target_names)-1`. Add these metadata values:

```python
"fit_scope": SEARCH_FIT_SCOPE,
"fit_document_count": matrix.shape[0],
"category_count": len(target_names),
```

During load, after version and required-field checks, enforce:

```python
if metadata["fit_scope"] != SEARCH_FIT_SCOPE:
    raise ValueError("artifact fit scope is outdated; rebuild with `python main.py build`")
if int(metadata["fit_document_count"]) != shape[0]:
    raise ValueError("artifact fit document count mismatch; rebuild with `python main.py build`")
if int(metadata["category_count"]) != len(target_names):
    raise ValueError("artifact category count mismatch; rebuild with `python main.py build`")
if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= len(target_names)):
    raise ValueError("artifact labels are outside target names; rebuild with `python main.py build`")
```

Place checks only after their required values (`shape`, `target_names`, `labels`) have been parsed.

- [ ] **Step 4: Run artifact and CLI-search regression tests**

Run: `.venv/bin/pytest tests/test_artifacts.py tests/test_cli.py tests/test_search.py -q`

Expected: PASS; search result fields and CLI error behavior remain unchanged.

- [ ] **Step 5: Run Ruff and commit the breaking artifact format**

Run: `.venv/bin/ruff check src/document_system/artifacts.py tests/test_artifacts.py`

Expected: `All checks passed!`

Commit:

```bash
git add src/document_system/artifacts.py tests/test_artifacts.py
git -c user.name=shannonlee-dev commit -m "feat(artifacts)!: require full-corpus search indexes" -m "BREAKING CHANGE: runtime artifact v1 must be rebuilt as v2."
```

---

### Task 3: Split classification and search build stages

**Files:**
- Create: `src/document_system/build_stages.py`
- Create: `tests/test_build_stages.py`
- Modify: `src/document_system/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `DatasetBundle`, `ClassificationReport`, `ValidationResult`, `SparseMatrix`, `NumpyTfidfVectorizer`
- Produces: `run_classification_stage(bundle, snippets, *, batch_size, epochs, random_state) -> ClassificationStageResult`
- Produces: `run_search_stage(bundle, snippets, queries, *, top_k=DEFAULT_TOP_K) -> SearchStageResult`
- Produces: `BuildReport` fields `classification_vocabulary_size`, `search_vocabulary_size`, `classification_validation_passed`, `search_validation_passed`, `classification_max_absolute_error`, `search_max_absolute_error`

- [ ] **Step 1: Write stage tests that expose held-out-only vocabulary behavior**

Create `tests/test_build_stages.py` with a small balanced bundle and fixed split:

```python
def fixed_split(*_args, **_kwargs):
    return np.arange(8), np.arange(8, 10)


def test_classification_and_search_use_independent_fit_scopes(
    monkeypatch, small_bundle
) -> None:
    created_vectorizers = []

    class RecordingVectorizer(NumpyTfidfVectorizer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created_vectorizers.append(self)

    monkeypatch.setattr(build_stages, "train_test_split", fixed_split)
    monkeypatch.setattr(
        build_stages, "NumpyTfidfVectorizer", RecordingVectorizer
    )
    snippets = tuple(make_safe_snippet(text) for text in small_bundle.texts)

    classification = build_stages.run_classification_stage(
        small_bundle,
        snippets,
        batch_size=2,
        epochs=1,
        random_state=42,
    )
    search = build_stages.run_search_stage(
        small_bundle,
        snippets,
        ("space orbit",),
    )

    classification_vectorizer, search_vectorizer = created_vectorizers
    assert classification_vectorizer is not search_vectorizer
    assert "heldoutonly" not in classification_vectorizer.vocabulary_
    assert "heldoutonly" in search_vectorizer.vocabulary_
    assert classification.validation.shape[0] == 8
    assert search.validation.shape[0] == 10
    assert search.matrix.shape[0] == 10
```

Build the fixture so rows 8 and 9 contain `heldoutonly` and rows 0–7 do not. The recording subclass is test-only; production result objects must not retain the classification vectorizer or its vocabulary.

- [ ] **Step 2: Run the stage test and verify import failure**

Run: `.venv/bin/pytest tests/test_build_stages.py -q`

Expected: FAIL because `document_system.build_stages` does not exist.

- [ ] **Step 3: Create focused result types and the classification stage**

Create `src/document_system/build_stages.py` with these result contracts:

```python
@dataclass(frozen=True)
class ClassificationStageResult:
    report: ClassificationReport
    validation: ValidationResult
    stage_example: dict[str, object]
    train_count: int
    test_count: int
    vocabulary_size: int


@dataclass(frozen=True)
class SearchStageResult:
    vectorizer: NumpyTfidfVectorizer
    matrix: SparseMatrix
    validation: ValidationResult
    search_examples: list[dict[str, object]]
    vocabulary_size: int
```

Implement classification with the existing calls in this order:

```python
def run_classification_stage(
    bundle: DatasetBundle,
    snippets: Sequence[str],
    *,
    batch_size: int,
    epochs: int,
    random_state: int,
) -> ClassificationStageResult:
    row_ids = np.arange(len(bundle.texts))
    train_ids, test_ids = train_test_split(
        row_ids,
        test_size=DEFAULT_TEST_SIZE,
        stratify=bundle.labels,
        random_state=random_state,
    )
    train_texts = [bundle.texts[int(row_id)] for row_id in train_ids]
    test_texts = [bundle.texts[int(row_id)] for row_id in test_ids]
    vectorizer = NumpyTfidfVectorizer(EnglishPreprocessor())
    stages = vectorizer.fit_transform_with_stages(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    validation = validate_against_sklearn(train_texts, vectorizer, stages.tfidf)
    if not validation.passed:
        raise RuntimeError(
            f"classification TF-IDF validation failed: {validation.max_absolute_error}"
        )
    model = train_linear_svm(
        stages.tfidf,
        bundle.labels[train_ids],
        batch_size=batch_size,
        epochs=epochs,
        random_state=random_state,
    )
    report = evaluate_classifier(
        model,
        test_matrix,
        bundle.labels[test_ids],
        [snippets[int(row_id)] for row_id in test_ids],
        bundle.target_names,
        batch_size=batch_size,
        document_ids=bundle.source_doc_ids[test_ids],
    )
    return ClassificationStageResult(
        report=report,
        validation=validation,
        stage_example=stage_example(stages, vectorizer),
        train_count=len(train_ids),
        test_count=len(test_ids),
        vocabulary_size=len(vectorizer.vocabulary_),
    )
```

- [ ] **Step 4: Implement the independent full-corpus search stage**

Use a new vectorizer and validate the complete matrix:

```python
def run_search_stage(
    bundle: DatasetBundle,
    snippets: Sequence[str],
    queries: Sequence[str],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> SearchStageResult:
    vectorizer = NumpyTfidfVectorizer(EnglishPreprocessor())
    matrix = vectorizer.fit_transform(bundle.texts)
    validation = validate_against_sklearn(bundle.texts, vectorizer, matrix)
    if not validation.passed:
        raise RuntimeError(
            f"search TF-IDF validation failed: {validation.max_absolute_error}"
        )
    searcher = DocumentSearch(
        vectorizer=vectorizer,
        matrix=matrix,
        snippets=snippets,
        labels=bundle.labels,
        target_names=bundle.target_names,
        document_ids=bundle.source_doc_ids,
    )
    examples = [
        {
            "query": query,
            "results": [
                result.to_dict()
                for result in searcher.search(query, min(top_k, len(bundle.texts)))
            ],
        }
        for query in queries
    ]
    return SearchStageResult(
        vectorizer=vectorizer,
        matrix=matrix,
        validation=validation,
        search_examples=examples,
        vocabulary_size=len(vectorizer.vocabulary_),
    )
```

- [ ] **Step 5: Run stage tests**

Run: `.venv/bin/pytest tests/test_build_stages.py -q`

Expected: PASS and prove the held-out token is absent only from the classification vocabulary.

- [ ] **Step 6: Write failing pipeline report tests for both feature spaces**

Update `tests/test_pipeline.py` to monkeypatch `document_system.build_stages.train_test_split` with the fixed 8/2 split and assert:

```python
assert report.classification_vocabulary_size < report.search_vocabulary_size
assert report.classification_validation_passed is True
assert report.search_validation_passed is True

validation = json.loads(
    (config.reports_dir / "tfidf_validation.json").read_text(encoding="utf-8")
)
assert validation["classification"]["fit_scope"] == "train_split"
assert validation["classification"]["fit_document_count"] == 8
assert validation["search"]["fit_scope"] == "full_corpus"
assert validation["search"]["fit_document_count"] == 10

metadata = json.loads(
    (config.runtime_dir / "metadata.json").read_text(encoding="utf-8")
)
assert "heldoutonly" in metadata["feature_names"]
assert metadata["fit_document_count"] == 10
```

Expected initial failure: `BuildReport` still has one vocabulary and one validation result.

- [ ] **Step 7: Reduce `pipeline.py` to orchestration and scoped report writing**

Replace the inline fit/split/model/search flow with sequential stage calls:

```python
snippets = tuple(make_safe_snippet(text) for text in bundle.texts)
classification = run_classification_stage(
    bundle,
    snippets,
    batch_size=config.batch_size,
    epochs=config.epochs,
    random_state=config.random_state,
)
search = run_search_stage(bundle, snippets, config.search_queries)
```

Define the explicit `BuildReport` fields from the spec and write scoped validation payloads:

```python
def _validation_payload(
    validation: ValidationResult,
    *,
    fit_scope: str,
    fit_document_count: int,
) -> dict[str, object]:
    payload = validation.to_dict()
    payload.update(
        {"fit_scope": fit_scope, "fit_document_count": fit_document_count}
    )
    return payload
```

Write `tfidf_validation.json` as:

```python
{
    "classification": _validation_payload(
        classification.validation,
        fit_scope="train_split",
        fit_document_count=classification.train_count,
    ),
    "search": _validation_payload(
        search.validation,
        fit_scope="full_corpus",
        fit_document_count=len(bundle.texts),
    ),
}
```

Add `classification_vocabulary_size` to `metrics.json`; add `fit_scope`, `fit_document_count`, `category_count` and `search_vocabulary_size` to `matrix_stats.json`; wrap `stage_example.json` with classification fit scope/count. Save runtime artifacts using `search.vectorizer` and `search.matrix`. Save the confusion matrix and misclassifications from `classification.report`.

Build the stage-example payload without changing its existing term/formula fields:

```python
stage_payload = dict(classification.stage_example)
stage_payload.update(
    {
        "fit_scope": "train_split",
        "fit_document_count": classification.train_count,
    }
)
```

- [ ] **Step 8: Run pipeline regression tests and Ruff**

Run: `.venv/bin/pytest tests/test_build_stages.py tests/test_pipeline.py tests/test_validation.py tests/test_classification.py tests/test_search.py -q`

Expected: PASS.

Run: `.venv/bin/ruff check src/document_system/build_stages.py src/document_system/pipeline.py tests/test_build_stages.py tests/test_pipeline.py`

Expected: `All checks passed!`

- [ ] **Step 9: Commit the dual-stage pipeline**

```bash
git add src/document_system/build_stages.py src/document_system/pipeline.py tests/test_build_stages.py tests/test_pipeline.py
git -c user.name=shannonlee-dev commit -m "feat(pipeline): separate classification and search TF-IDF spaces"
```

---

### Task 4: Expose dual build results and update ablation semantics

**Files:**
- Modify: `src/document_system/cli.py`
- Modify: `src/document_system/ablation.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Consumes: new `BuildReport` fields from Task 3
- Preserves: `main.py build` and search CLI argument syntax
- Produces: build output labels `classification_vocabulary`, `search_vocabulary`, `classification_max_error`, `search_max_error`
- Produces: ablation report dataset value `20 Newsgroups: all 20 categories`

- [ ] **Step 1: Write failing CLI output and ablation-description tests**

In `tests/test_cli.py`, monkeypatch `document_system.pipeline.build_project` to return a `SimpleNamespace` containing every field consumed by `_run_build`, then assert:

```python
exit_code = main(["build"])
output = capsys.readouterr().out
assert exit_code == 0
assert "classification_vocabulary=101" in output
assert "search_vocabulary=109" in output
assert "classification_max_error=1.000e-08" in output
assert "search_max_error=2.000e-08" in output
```

In `tests/test_ablation.py`, run `run_stop_word_ablation()` with a small 20-document/two-label `DatasetBundle` and assert:

```python
assert report["dataset"] == "20 Newsgroups: all 20 categories"
assert report["split"]["train_documents"] == 16
assert report["split"]["test_queries"] == 4
```

Use ten documents per label so the 80% training corpus has at least the required retrieval `k=10`.

- [ ] **Step 2: Run focused tests and verify text mismatches**

Run: `.venv/bin/pytest tests/test_cli.py tests/test_ablation.py -q`

Expected: FAIL because CLI reads removed singular fields and ablation still names three categories.

- [ ] **Step 3: Update CLI output and ablation dataset name**

Change the build completion message to:

```python
print(
    f"build complete: documents={report.document_count:,}, "
    f"classification_vocabulary={report.classification_vocabulary_size:,}, "
    f"search_vocabulary={report.search_vocabulary_size:,}, "
    f"classification_max_error={report.classification_max_absolute_error:.3e}, "
    f"search_max_error={report.search_max_absolute_error:.3e}, "
    f"accuracy={report.accuracy:.4f}, macro_f1={report.macro_f1:.4f}"
)
```

In `run_stop_word_ablation()`, change only the dataset description string to `20 Newsgroups: all 20 categories`. Preserve its 80/20 fit, relevance definition and computation.

- [ ] **Step 4: Run focused and combined tests**

Run: `.venv/bin/pytest tests/test_cli.py tests/test_ablation.py tests/test_pipeline.py tests/test_artifacts.py -q`

Expected: PASS.

Run: `.venv/bin/ruff check src/document_system/cli.py src/document_system/ablation.py tests/test_cli.py tests/test_ablation.py`

Expected: `All checks passed!`

- [ ] **Step 5: Commit observable build output changes**

```bash
git add src/document_system/cli.py src/document_system/ablation.py tests/test_cli.py tests/test_ablation.py
git -c user.name=shannonlee-dev commit -m "feat(cli): report both TF-IDF build spaces"
```

---

### Task 5: Verify the complete code change before expensive dataset work

**Files:**
- Verify only: `src/`, `tests/`

**Interfaces:**
- Consumes: Tasks 1–4
- Produces: green unit-test and static-analysis evidence before dataset build

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/pytest -q`

Expected: PASS with no failures or skips introduced by this change.

- [ ] **Step 2: Run full Ruff validation**

Run: `.venv/bin/ruff check src tests`

Expected: `All checks passed!`

- [ ] **Step 3: Inspect repository-wide stale code assumptions**

Run:

```bash
rg -n "SAFE_CATEGORIES|report\.vocabulary_size|report\.validation_passed|report\.max_absolute_error" src tests
```

Expected: no matches.

Run:

```bash
rg -n "categories=SAFE_CATEGORIES|20 Newsgroups: comp\.graphics" src tests
```

Expected: no matches.

- [ ] **Step 4: Inspect the cumulative implementation diff**

Run: `git diff HEAD~4 --check`

Expected: no whitespace errors.

Run: `git diff HEAD~4 --stat`

Expected: changes are limited to the dataset, artifacts, stages, pipeline, CLI, ablation and their tests.

No commit is created in this task because it is a verification gate over already committed code.

---

### Task 6: Regenerate 20-category evidence and align documentation

**Files:**
- Modify: `README.md`
- Modify: `DATASET_LICENSE.md`
- Modify: `docs/stop-word-ablation.md`
- Delete: `artifacts/reports/stop_word_ablation.json`
- Regenerate: `artifacts/reports/confusion_matrix.png`
- Regenerate: `artifacts/reports/matrix_stats.json`
- Regenerate: `artifacts/reports/metrics.json`
- Regenerate: `artifacts/reports/misclassifications.json`
- Regenerate: `artifacts/reports/privacy_report.json`
- Regenerate: `artifacts/reports/search_examples.json`
- Regenerate: `artifacts/reports/stage_example.json`
- Regenerate: `artifacts/reports/tfidf_validation.json`
- Runtime-only: `artifacts/runtime/matrix.npz`, `artifacts/runtime/metadata.json`

**Interfaces:**
- Consumes: `python main.py build` from Tasks 1–4
- Produces: repository documentation and reports consistent with the actual 20-category build
- Preserves: runtime files remain ignored and are not committed

- [ ] **Step 1: Run the actual full build**

Run: `.venv/bin/python main.py build`

Expected: exit code 0; output includes `documents`, both vocabulary sizes, both maximum errors, Accuracy and macro F1. The first uncached run may download the public dataset through scikit-learn.

- [ ] **Step 2: Validate generated report semantics programmatically**

Run:

```bash
.venv/bin/python -c 'import json; from pathlib import Path; p=Path("artifacts/reports"); m=json.loads((p/"metrics.json").read_text()); s=json.loads((p/"matrix_stats.json").read_text()); v=json.loads((p/"tfidf_validation.json").read_text()); q=json.loads((p/"privacy_report.json").read_text()); assert m["category_count"] == 20; assert m["train_count"] + m["test_count"] == m["document_count"]; assert s["fit_scope"] == "full_corpus"; assert s["fit_document_count"] == m["document_count"]; assert q["documents_retained"] == m["document_count"]; assert len(q["category_counts"]) == 20; assert v["classification"]["passed"] and v["search"]["passed"]; assert v["classification"]["max_absolute_error"] <= 1e-6; assert v["search"]["max_absolute_error"] <= 1e-6'
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Smoke-test v2 runtime loading and search**

Run: `.venv/bin/python main.py --query "space shuttle orbit" --topk 5`

Expected: exit code 0 and five ranked results with score, document ID, category and bounded snippet.

Run:

```bash
.venv/bin/python -c 'from document_system.artifacts import load_search_artifacts; a=load_search_artifacts("artifacts/runtime"); assert a.matrix.shape[0] == len(a.snippets) == a.labels.size == a.document_ids.size; assert len(a.target_names) == 20'
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Remove stale three-category ablation evidence**

Run: `git rm artifacts/reports/stop_word_ablation.json`

The deletion is intentional and recoverable from Git history. Do not run the current quadratic retrieval ablation over all 20 categories as part of this task.

- [ ] **Step 5: Update documentation with measured values**

In `README.md`:

- change the architecture diagram to branch after sanitization into classification 80/20 and search 100% fit paths;
- replace the three-category selection section with the fixed 20-category contract;
- explain that classification and search use independent vocabulary/IDF spaces;
- copy exact document counts, train/test counts, both vocabulary sizes, matrix shape, Accuracy, macro F1 and both validation errors from generated JSON;
- replace the three-class confusion-matrix description with the generated 20x20 result;
- remove the current three-category ablation deltas and link to the revised method/limitation note;
- state that `sci.med` is included and that current structured redaction does not remove free-form health information.

In `DATASET_LICENSE.md`, replace “세 카테고리만 선택” with the complete 20-category range and record the accepted residual privacy risk without claiming PII-free handling.

In `docs/stop-word-ablation.md`, document that the method now consumes the full-category loader but its Python query-by-corpus loop is quadratic; explain that the previous three-category JSON was removed and no new numerical comparison is claimed until that workflow is run or separately optimized.

- [ ] **Step 6: Check documentation and artifact consistency**

Run:

```bash
rg -n "세 카테고리만|SAFE_CATEGORIES|현재 3-category|2,863|21,662" README.md DATASET_LICENSE.md docs/stop-word-ablation.md artifacts/reports
```

Expected: no stale current-state claims. A clearly labeled historical comparison may retain old values only if its heading and surrounding text state that it is historical.

Run:

```bash
git status --short --ignored artifacts/runtime artifacts/reports
```

Expected: runtime files are ignored; report changes are visible and `stop_word_ablation.json` is deleted.

- [ ] **Step 7: Run final tests and static checks after documentation updates**

Run: `.venv/bin/pytest -q`

Expected: PASS.

Run: `.venv/bin/ruff check src tests`

Expected: `All checks passed!`

- [ ] **Step 8: Inspect and commit documentation plus generated evidence**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git diff --stat`

Confirm that ignored `artifacts/runtime/*` is absent and only the listed docs/reports are staged.

Commit:

```bash
git add README.md DATASET_LICENSE.md docs/stop-word-ablation.md artifacts/reports
git -c user.name=shannonlee-dev commit -m "docs: publish 20-category build evidence"
```

---

## Final Verification

- [ ] Run `.venv/bin/pytest -q` and confirm all tests pass.
- [ ] Run `.venv/bin/ruff check src tests` and confirm `All checks passed!`.
- [ ] Run `.venv/bin/python main.py --query "graphics image rendering" --topk 5` against v2 runtime artifacts.
- [ ] Confirm `git diff --check HEAD~5` reports no whitespace errors.
- [ ] Confirm `git status --short --branch` is clean on `feature/20-categories-full-search-matrix`.
- [ ] Review `git log --oneline --decorate -n 7` and verify one coherent commit per feature boundary.
