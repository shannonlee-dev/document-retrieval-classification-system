# Document Retrieval and Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible CLI that uses all 18,846 20 Newsgroups documents for NumPy-only sparse TF-IDF, cosine retrieval, and linear-SVM classification, with written evaluation evidence.

**Architecture:** A small `src/document_system` package owns preprocessing, a CSR-like NumPy container, TF-IDF fitting/transformation, validation, retrieval, classification, and artifact I/O. The full corpus remains sparse; classification connects the same NumPy buffers to Scikit-learn through a CSR container. `main.py` exposes build and search workflows, and generated report artifacts supply every numeric claim in README.

**Tech Stack:** Python 3.10+, NumPy, Scikit-learn, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-08-25-document-retrieval-design.md`

## Global Constraints

- Use all 18,846 documents and all 20 categories; do not sample documents or cap vocabulary with `max_features`.
- Split the full dataset stratified 8:2 with `random_state=42`.
- Implement TF, smoothed IDF, TF-IDF, L2 normalization, and cosine similarity with NumPy; Scikit-learn is for validation and classification.
- Match `smooth_idf=True`, `sublinear_tf=False`, `norm="l2"`, `use_idf=True`, and `dtype=float64` within `1e-6`.
- Keep the full TF-IDF corpus sparse and justify it with measured `nnz`, density, sparsity, and byte counts.
- Support `python main.py --query "space shuttle orbit" --topk 5` after `python main.py build`.
- Record actual results rather than presenting example values as measured results.

---

### Task 1: Project Foundation, Dataset, and Preprocessing

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/document_system/__init__.py`
- Create: `src/document_system/preprocessing.py`
- Create: `src/document_system/dataset.py`
- Create: `tests/test_preprocessing.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Produces: `EnglishPreprocessor.tokenize(text: str) -> list[str]`
- Produces: `DatasetBundle(texts, labels, target_names)` and `load_20newsgroups() -> DatasetBundle`
- Produces: `validate_dataset(texts, labels, *, minimum_documents=500) -> None`

- [ ] **Step 1: Write failing preprocessing and dataset tests**

```python
def test_tokenize_normalizes_and_filters():
    processor = EnglishPreprocessor(stop_words={"the", "and"})
    assert processor.tokenize("The QUICK, brown fox 123 and x!") == ["quick", "brown", "fox"]

def test_validate_dataset_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        validate_dataset(["one", "two"], [0], minimum_documents=2)

def test_validate_dataset_requires_two_labels():
    with pytest.raises(ValueError, match="two labels"):
        validate_dataset(["one", "two"], [0, 0], minimum_documents=2)
```

- [ ] **Step 2: Run tests and confirm missing-module failures**

Run: `python -m pytest tests/test_preprocessing.py tests/test_dataset.py -q`

Expected: collection fails because `document_system.preprocessing` and `document_system.dataset` do not exist.

- [ ] **Step 3: Add package configuration and dependencies**

Use a `src` package layout, require Python `>=3.10`, declare NumPy, Scikit-learn, and Matplotlib runtime dependencies, and configure pytest with `pythonpath = ["src"]`. Put the same runtime packages in `requirements.txt` because the mission explicitly requires that file. Ignore caches, virtual environments, downloaded dataset data, and `artifacts/runtime/`, while keeping `artifacts/reports/` trackable.

- [ ] **Step 4: Implement deterministic preprocessing and dataset validation**

```python
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")

@dataclass(frozen=True)
class EnglishPreprocessor:
    stop_words: frozenset[str] = DEFAULT_STOP_WORDS

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [
            token for token in TOKEN_PATTERN.findall(text.lower())
            if len(token) > 1 and token not in self.stop_words
        ]
```

`load_20newsgroups()` must call `fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), shuffle=True, random_state=42)` and validate exactly 18,846 documents and 20 target names.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_preprocessing.py tests/test_dataset.py -q`

Expected: all focused tests pass without downloading the dataset.

- [ ] **Step 6: Commit the foundation**

```bash
git add pyproject.toml requirements.txt .gitignore src/document_system tests/test_preprocessing.py tests/test_dataset.py
git commit -m "feat: add dataset and preprocessing foundation"
```

### Task 2: NumPy Sparse Matrix and TF-IDF

**Files:**
- Create: `src/document_system/sparse_matrix.py`
- Create: `src/document_system/tfidf.py`
- Create: `tests/test_sparse_matrix.py`
- Create: `tests/test_tfidf.py`

**Interfaces:**
- Consumes: `EnglishPreprocessor.tokenize(text)`
- Produces: `SparseMatrix(data, indices, indptr, shape)` with `row()`, `to_dense_rows()`, `memory_stats()`
- Produces: `NumpyTfidfVectorizer.fit`, `transform`, and `fit_transform`
- Produces: `TfidfStages(counts, idf, tfidf)` for stage inspection

- [ ] **Step 1: Write failing sparse-matrix tests**

```python
def test_sparse_matrix_restores_selected_rows():
    matrix = SparseMatrix(
        data=np.array([2.0, 1.0, 3.0]),
        indices=np.array([0, 2, 1], dtype=np.int32),
        indptr=np.array([0, 2, 3], dtype=np.int32),
        shape=(2, 3),
    )
    np.testing.assert_array_equal(matrix.to_dense_rows([1]), [[0.0, 3.0, 0.0]])

def test_memory_stats_use_actual_numpy_bytes():
    matrix = SparseMatrix(
        data=np.array([2.0, 1.0, 3.0]),
        indices=np.array([0, 2, 1], dtype=np.int32),
        indptr=np.array([0, 2, 3], dtype=np.int32),
        shape=(2, 3),
    )
    stats = matrix.memory_stats()
    assert stats["dense_bytes"] == 2 * 3 * 8
    assert stats["sparse_bytes"] == sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr))
```

- [ ] **Step 2: Run sparse tests and confirm failure**

Run: `python -m pytest tests/test_sparse_matrix.py -q`

Expected: FAIL because `SparseMatrix` does not exist.

- [ ] **Step 3: Implement the validated sparse container**

Validate one-dimensional arrays, monotonic `indptr`, matching nonzero counts, sorted in-range column indices per row, and shape consistency. `row(row_id)` returns views of indices and data. `to_dense_rows(row_ids)` allocates only the requested rows. `memory_stats()` returns shape, `nnz`, density, sparsity, dense bytes, sparse bytes, and compression ratio.

- [ ] **Step 4: Write failing TF-IDF stage and normalization tests**

```python
def test_tfidf_exposes_raw_tf_smoothed_idf_and_l2_values():
    texts = ["apple apple banana", "banana carrot"]
    vectorizer = NumpyTfidfVectorizer(EnglishPreprocessor(stop_words=frozenset()))
    stages = vectorizer.fit_transform_with_stages(texts)
    apple = vectorizer.vocabulary_["apple"]
    banana = vectorizer.vocabulary_["banana"]
    assert stages.counts.to_dense_rows([0])[0, apple] == 2
    assert stages.counts.to_dense_rows([0])[0, banana] == 1
    np.testing.assert_allclose(vectorizer.idf_[apple], np.log(3 / 2) + 1)
    np.testing.assert_allclose(np.linalg.norm(stages.tfidf.to_dense_rows([0])[0]), 1.0)

def test_transform_ignores_out_of_vocabulary_terms_and_keeps_zero_row():
    vectorizer = NumpyTfidfVectorizer(EnglishPreprocessor(stop_words=frozenset())).fit(["known term"])
    transformed = vectorizer.transform(["unknown"])
    assert transformed.nnz == 0
    np.testing.assert_array_equal(transformed.to_dense_rows([0]), [[0.0, 0.0]])
```

- [ ] **Step 5: Run TF-IDF tests and confirm failure**

Run: `python -m pytest tests/test_tfidf.py -q`

Expected: FAIL because `NumpyTfidfVectorizer` does not exist.

- [ ] **Step 6: Implement vocabulary, raw counts, IDF, and normalization**

Build a sorted vocabulary from training tokens. For each row, use `collections.Counter`, sort vocabulary indices, and append raw counts to the three NumPy-array buffers. Compute document frequency by counting each stored column once per row. Apply `idf = log((1 + n_documents) / (1 + document_frequency)) + 1`, multiply `data` by `idf[indices]`, and L2-normalize each row without changing zero rows.

- [ ] **Step 7: Run focused TF-IDF tests**

Run: `python -m pytest tests/test_sparse_matrix.py tests/test_tfidf.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit NumPy TF-IDF**

```bash
git add src/document_system/sparse_matrix.py src/document_system/tfidf.py tests/test_sparse_matrix.py tests/test_tfidf.py
git commit -m "feat: implement NumPy sparse TF-IDF"
```

### Task 3: Scikit-learn Validation and Mathematical Evidence

**Files:**
- Create: `src/document_system/validation.py`
- Create: `tests/test_validation.py`

**Interfaces:**
- Consumes: fitted `NumpyTfidfVectorizer`, original texts, and custom `SparseMatrix`
- Produces: `validate_against_sklearn(...) -> ValidationResult`
- Produces: `stage_example(...) -> dict[str, object]`

- [ ] **Step 1: Write failing equivalence tests**

```python
def test_numpy_tfidf_matches_sklearn_without_dense_full_matrix():
    texts = ["apple apple banana", "banana carrot", "carrot date"]
    processor = EnglishPreprocessor(stop_words=frozenset())
    vectorizer = NumpyTfidfVectorizer(processor)
    custom = vectorizer.fit_transform(texts)
    result = validate_against_sklearn(texts, vectorizer, custom)
    assert result.passed is True
    assert result.max_absolute_error <= 1e-6
    assert result.settings == {
        "smooth_idf": True,
        "sublinear_tf": False,
        "norm": "l2",
        "use_idf": True,
        "dtype": "float64",
    }
```

- [ ] **Step 2: Run validation test and confirm failure**

Run: `python -m pytest tests/test_validation.py -q`

Expected: FAIL because the validation module does not exist.

- [ ] **Step 3: Implement sparse row-wise comparison**

Configure `TfidfVectorizer` with the custom fixed vocabulary and an analyzer that calls the same `EnglishPreprocessor.tokenize`. Compare each custom row against sklearn CSR row indices and values. Treat missing entries on either side as zero, accumulate maximum error, total absolute error, and compared element count, and raise if shape or vocabulary differs.

- [ ] **Step 4: Add stage evidence generation**

`stage_example()` selects the first nonzero document and up to five terms, returning raw TF, IDF, unnormalized TF-IDF, and normalized TF-IDF values together with formulas and vocabulary indices. Values must come from actual arrays rather than duplicated calculations in README.

- [ ] **Step 5: Run focused and regression tests**

Run: `python -m pytest tests/test_validation.py tests/test_tfidf.py -q`

Expected: all tests pass and maximum error is below the threshold.

- [ ] **Step 6: Commit validation**

```bash
git add src/document_system/validation.py tests/test_validation.py
git commit -m "test: validate NumPy TF-IDF against sklearn"
```

### Task 4: Sparse Cosine Search

**Files:**
- Create: `src/document_system/search.py`
- Create: `tests/test_search.py`

**Interfaces:**
- Consumes: fitted vectorizer, normalized document matrix, document texts, labels, and target names
- Produces: `sparse_dot(left_indices, left_data, right_indices, right_data) -> float`
- Produces: `DocumentSearch.search(query: str, topk: int = 5) -> list[SearchResult]`

- [ ] **Step 1: Write failing dot-product and ranking tests**

```python
def test_sparse_dot_equals_dense_dot():
    left_i = np.array([0, 3], dtype=np.int32)
    right_i = np.array([1, 3], dtype=np.int32)
    assert sparse_dot(left_i, np.array([0.5, 0.7]), right_i, np.array([0.2, 0.4])) == pytest.approx(0.28)

def test_search_returns_score_id_label_and_snippet():
    searcher = make_small_searcher(["space shuttle orbit", "baseball pitcher game"])
    result = searcher.search("shuttle orbit", topk=1)[0]
    assert result.doc_id == 0
    assert result.score > 0
    assert result.text_snippet.startswith("space shuttle")

def test_search_rejects_oov_query():
    with pytest.raises(ValueError, match="vocabulary"):
        searcher.search("zzzzunknown", topk=5)
```

- [ ] **Step 2: Run search tests and confirm failure**

Run: `python -m pytest tests/test_search.py -q`

Expected: FAIL because the search module does not exist.

- [ ] **Step 3: Implement exact sparse cosine search**

Use a two-pointer intersection over sorted column indices for each dot product. Because rows and query are L2-normalized, use the dot directly as cosine similarity. Validate `1 <= topk <= document_count`, reject empty/OOV queries, and rank with `np.lexsort((doc_ids, -scores))` so equal scores resolve by ascending document ID.

- [ ] **Step 4: Run search tests and all mathematical tests**

Run: `python -m pytest tests/test_search.py tests/test_sparse_matrix.py tests/test_tfidf.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit search**

```bash
git add src/document_system/search.py tests/test_search.py
git commit -m "feat: add NumPy cosine document search"
```

### Task 5: Batched Linear-SVM Classification and Evaluation

**Files:**
- Create: `src/document_system/classification.py`
- Create: `tests/test_classification.py`

**Interfaces:**
- Consumes: train/test `SparseMatrix`, NumPy labels, and target names
- Produces: `to_sklearn_csr(matrix) -> scipy.sparse.csr_matrix` and `train_linear_svm(...) -> SGDClassifier`
- Produces: `evaluate_classifier(...) -> ClassificationReport`
- Produces: `save_confusion_matrix(report, target_names, path) -> None`

- [ ] **Step 1: Write failing batched-classification tests**

```python
def test_linear_svm_trains_from_sparse_batches_reproducibly():
    matrix = small_separable_tfidf_matrix()
    labels = np.array([0, 0, 1, 1])
    first = train_linear_svm(matrix, labels, batch_size=2, epochs=4, random_state=42)
    second = train_linear_svm(matrix, labels, batch_size=2, epochs=4, random_state=42)
    np.testing.assert_array_equal(first.predict(matrix.to_dense_rows(range(4))), second.predict(matrix.to_dense_rows(range(4))))

def test_evaluation_contains_accuracy_macro_f1_confusion_and_errors():
    report = evaluate_classifier(model, test_matrix, labels, texts, target_names)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0
    assert report.confusion_matrix.shape == (2, 2)
    assert all({"doc_id", "actual", "predicted", "text_snippet"} <= item.keys() for item in report.misclassifications)
```

- [ ] **Step 2: Run classification tests and confirm failure**

Run: `python -m pytest tests/test_classification.py -q`

Expected: FAIL because the classification module does not exist.

- [ ] **Step 3: Implement CSR batches and hinge-loss SGD**

For each epoch, shuffle row IDs with `np.random.default_rng(random_state)` and call `partial_fit` with CSR row batches; pass all classes on the first call. Construct the CSR container directly over the custom NumPy `data`, `indices`, and `indptr` arrays without recomputing TF-IDF. Configure `SGDClassifier(loss="hinge", random_state=42, max_iter=1, tol=None, shuffle=False)`. Prediction also proceeds in sparse batches.

- [ ] **Step 4: Implement metrics, misclassification capture, and confusion image**

Use sklearn metrics for Accuracy, macro F1, and confusion matrix. Capture at least the first five errors after deterministic test ordering, including actual/predicted label names and a normalized 240-character snippet. Render a labeled 20x20 Matplotlib image with readable rotation and save it without opening a GUI.

- [ ] **Step 5: Run classification tests**

Run: `python -m pytest tests/test_classification.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit classification**

```bash
git add src/document_system/classification.py tests/test_classification.py
git commit -m "feat: add batched linear SVM evaluation"
```

### Task 6: Artifact Persistence and CLI Pipeline

**Files:**
- Create: `src/document_system/artifacts.py`
- Create: `src/document_system/pipeline.py`
- Create: `src/document_system/cli.py`
- Create: `main.py`
- Create: `tests/test_artifacts.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: every earlier module
- Produces: `build_project(config: BuildConfig) -> BuildReport`
- Produces: `save_search_artifacts(...)` and `load_search_artifacts(...)`
- Produces: CLI `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing round-trip and CLI tests**

```python
def test_search_artifacts_round_trip(tmp_path):
    save_search_artifacts(tmp_path, vectorizer, matrix, texts, labels, target_names)
    restored = load_search_artifacts(tmp_path)
    assert restored.matrix.shape == matrix.shape
    assert restored.vectorizer.vocabulary_ == vectorizer.vocabulary_
    assert restored.texts == texts

def test_cli_search_prints_rank_score_id_and_snippet(tmp_path, capsys):
    write_small_artifacts(tmp_path)
    exit_code = main(["--artifacts", str(tmp_path), "--query", "space orbit", "--topk", "1"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "1." in output and "score=" in output and "doc_id=" in output

def test_cli_missing_artifacts_explains_build_command(tmp_path, capsys):
    exit_code = main(["--artifacts", str(tmp_path), "--query", "space"])
    assert exit_code == 2
    assert "python main.py build" in capsys.readouterr().err
```

- [ ] **Step 2: Run artifact and CLI tests and confirm failure**

Run: `python -m pytest tests/test_artifacts.py tests/test_cli.py -q`

Expected: FAIL because persistence and CLI modules do not exist.

- [ ] **Step 3: Implement versioned runtime artifacts**

Store sparse arrays and IDF with `np.savez_compressed`, vocabulary/configuration as UTF-8 JSON, and texts/labels/target names as UTF-8 JSON. Include artifact version, shape, dtype, and preprocessing settings. Reject missing files, version mismatch, shape mismatch, and vocabulary/IDF length mismatch with a rebuild message.

- [ ] **Step 4: Implement the full build pipeline**

Load all data, split indices with `train_test_split(test_size=0.2, stratify=labels, random_state=42)`, fit only on training texts, transform test and full corpus, validate the training matrix, train/evaluate the classifier, run three fixed search queries, and write runtime plus report artifacts. Assert document and category counts before writing success logs.

- [ ] **Step 5: Implement exact CLI forms**

`python main.py build` accepts artifact directory, batch size, and epochs with documented defaults. `python main.py --query TEXT --topk N` loads the index and prints rank, six-decimal score, document ID, category, and snippet. Convert expected user errors to exit code 2 without a traceback; unexpected errors remain visible.

- [ ] **Step 6: Run CLI and full unit suite**

Run: `python -m pytest -q`

Expected: all tests pass without requiring a network download.

- [ ] **Step 7: Commit the runnable pipeline**

```bash
git add main.py src/document_system/artifacts.py src/document_system/pipeline.py src/document_system/cli.py tests/test_artifacts.py tests/test_cli.py
git commit -m "feat: add reproducible build and search CLI"
```

### Task 7: Full-Corpus Evidence and README

**Files:**
- Modify: `README.md`
- Create: `artifacts/reports/tfidf_sklearn_validation.json`
- Create: `artifacts/reports/search_index_statistics.json`
- Create: `artifacts/reports/classification_metrics.json`
- Create: `artifacts/reports/tfidf_transformation_example.json`
- Create: `artifacts/reports/classification_error_examples.json`
- Create: `artifacts/reports/search_result_examples.json`
- Create: `artifacts/reports/classification_confusion_matrix.png`
- Create: `tests/test_readme.py`

**Interfaces:**
- Consumes: `python main.py build` report artifacts
- Produces: submission-ready README and committed evaluation evidence

- [ ] **Step 1: Write failing README contract test**

```python
@pytest.mark.parametrize("heading", ["구현 요약", "검증/실험 결과", "한계와 개선 방향", "실행 방법"])
def test_readme_contains_required_sections(heading):
    assert heading in Path("README.md").read_text(encoding="utf-8")

def test_readme_records_required_commands_and_settings():
    readme = Path("README.md").read_text(encoding="utf-8")
    for text in ["python main.py build", "--query", "smooth_idf=True", "sublinear_tf=False", "norm='l2'"]:
        assert text in readme
```

- [ ] **Step 2: Run README test and confirm failure**

Run: `python -m pytest tests/test_readme.py -q`

Expected: FAIL because the current README lacks required sections and commands.

- [ ] **Step 3: Install the project and run the full corpus build**

Run: `python -m pip install -e '.[dev]'`

Run: `python main.py build`

Expected: dataset count 18,846; category count 20; TF-IDF validation PASS; report files and confusion matrix created. If network access is unavailable, rerun after obtaining permission rather than substituting fabricated values.

- [ ] **Step 4: Inspect generated evidence for internal consistency**

Verify that train plus test counts equal 18,846, matrix columns equal vocabulary size, validation error is at most `1e-6`, sparse bytes are below dense bytes, Accuracy and macro F1 are between 0 and 1, the confusion matrix is 20x20, at least five misclassifications exist, and each search query has five results.

- [ ] **Step 5: Rewrite README from measured artifacts**

Use the required three-part report structure. Include preprocessing rationale; TF/IDF/TF-IDF formulas; a stage table sourced from `tfidf_transformation_example.json`; matrix shape, `nnz`, density, sparsity, and byte comparison; sklearn settings and errors; the linear-SVM setup; Accuracy, macro F1, and the confusion image; five or more case-specific error analyses; actual Top-5 search output; BoW examples for word order and polysemy; cosine versus Euclidean distance; vocabulary row/column mapping; input portability contract; 1-million-document bottlenecks and indexing/sharding/ANN responses; and an embedding/BERT comparison plan with Accuracy, macro F1, mAP, and latency metrics.

- [ ] **Step 6: Verify README, CLI search, and full tests**

Run: `python -m pytest -q`

Run: `python main.py --query "space shuttle orbit" --topk 5`

Expected: tests pass and five ranked results print with score, document ID, category, and snippet.

- [ ] **Step 7: Self-review the complete submission diff**

Run: `git diff --check`

Run: `git status --short`

Confirm no dataset cache or runtime index is staged, no example value is represented as measured, and every private rubric item has a corresponding README section or report artifact.

- [ ] **Step 8: Commit reports and documentation**

```bash
git add README.md artifacts/reports tests/test_readme.py
git commit -m "docs: report full corpus retrieval results"
```
