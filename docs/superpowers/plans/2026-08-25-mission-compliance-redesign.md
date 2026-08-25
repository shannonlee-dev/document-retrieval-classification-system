# Mission Compliance Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace unsafe corpus handling and non-compliant numerical boundaries while preserving the existing build and search CLI.

**Architecture:** Sanitize selected public-source documents at the dataset boundary into a fixed safe vocabulary, carry stable source IDs separately, and persist only sanitized snippets. Keep the NumPy TF-IDF core, replace scalar cosine and direct SciPy adapters with NumPy operations, then rebuild all reports in place.

**Tech Stack:** Python 3.10+, NumPy, Scikit-learn for dataset loading/classification/TF-IDF validation, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-08-25-mission-compliance-redesign.md`

## Global Constraints

- Do not implement bonus TF variants, BM25, or an inverted index.
- Keep `python main.py build` and `python main.py --query "..." --topk 5`.
- Keep `ARTIFACT_VERSION = 1`; overwrite current runtime and report files.
- Do not import `scipy` directly from project or test code.
- Store only non-empty text composed of the fixed safe vocabulary.
- Keep TF-IDF maximum absolute validation error at or below `1e-6`.
- Use tests first for every production behavior change.

---

### Task 1: Privacy-safe dataset boundary and whitespace tokenization

**Files:**
- Create: `src/document_system/privacy.py`
- Create: `tests/test_privacy.py`
- Modify: `src/document_system/preprocessing.py`
- Modify: `src/document_system/dataset.py`
- Modify: `tests/test_preprocessing.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Produces: `SAFE_TERMS: frozenset[str]`
- Produces: `sanitize_text(text: str) -> str`
- Produces: `is_safe_text(text: str) -> bool`
- Produces: `DatasetBundle.source_doc_ids: np.ndarray`
- Preserves: `EnglishPreprocessor.tokenize(text: str) -> list[str]`

- [ ] **Step 1: Add failing privacy and whitespace-tokenizer tests**

Add tests that require these exact behaviors:

    def test_sanitize_text_keeps_only_safe_generic_terms() -> None:
        text = "Alice alice@example.com called 010-1234-5678 about treatment. Rocket orbit!"
        assert sanitize_text(text) == "rocket orbit"
        assert is_safe_text("rocket orbit")
        assert not is_safe_text("rocket alice")

    def test_whitespace_tokenizer_cleans_then_splits() -> None:
        processor = EnglishPreprocessor(stop_words=frozenset({"and"}))
        assert processor.tokenize("Image, PIXEL and orbit-rocket!") == [
            "image", "pixel", "orbit", "rocket"
        ]

    def test_validate_dataset_rejects_blank_text() -> None:
        with pytest.raises(ValueError, match="blank"):
            validate_dataset(["space", "   "], [0, 1], minimum_documents=2)

    def test_validate_dataset_checks_source_document_ids() -> None:
        with pytest.raises(ValueError, match="document IDs"):
            validate_dataset(
                ["space", "baseball"],
                [0, 1],
                source_doc_ids=[7],
                minimum_documents=2,
            )

    def test_loader_returns_only_safe_nonblank_documents(monkeypatch) -> None:
        raw_texts = (
            ["Alice alice@example.com image pixel"] * 200
            + ["Bob 010-1234-5678 baseball pitcher"] * 200
            + ["medical treatment details rocket orbit"] * 200
            + ["Alice alice@example.com treatment"]  # removed as blank
        )
        raw_labels = np.array([0] * 200 + [1] * 200 + [2] * 200 + [0])
        fake = SimpleNamespace(
            data=raw_texts,
            target=raw_labels,
            target_names=["comp.graphics", "rec.sport.baseball", "sci.space"],
        )
        monkeypatch.setattr(dataset_module, "fetch_20newsgroups", lambda **_: fake)

        bundle = dataset_module.load_20newsgroups()

        assert len(bundle.texts) == 600
        assert len(bundle.source_doc_ids) == 600
        assert all(is_safe_text(text) for text in bundle.texts)

- [ ] **Step 2: Run the new tests and verify RED**

Run:

    .venv/bin/python -m pytest tests/test_privacy.py tests/test_preprocessing.py tests/test_dataset.py -q

Expected: failures because `privacy.py`, whitespace behavior, blank rejection, and source ID validation do not exist.

- [ ] **Step 3: Implement the safe-text boundary**

Create `privacy.py` with a fixed allowlist covering generic graphics, baseball, and space vocabulary. Include at least the following exact terms and no person, contact, location, or medical terms:

    SAFE_TERMS = frozenset(
        """
        algorithm animation astronaut ball baseball bat batting bitmap catcher color
        computer data dimensional display earth field file flight format galaxy
        game graphics gravity hit hitter image inning launch league loss lunar
        mission model monitor moon object orbit orbital pitcher pixel planet
        player program render resolution rocket run satellite score screen
        season shuttle software solar space spacecraft stadium star team telescope
        throw tracing universe vector video visual win window
        """.split()
    )

Implement `sanitize_text` with `re.findall(r"[a-z]+", text.lower())`, retain only `SAFE_TERMS`, and join with one space. Implement `is_safe_text` so blank text is false and every whitespace token must be in the allowlist.

Change `EnglishPreprocessor.tokenize` to lowercase, replace `[^a-z\s]+` with spaces, split on whitespace, and apply existing length/stop-word rules.

Extend `validate_dataset` with optional `source_doc_ids: Sequence[int] | None = None`; reject blank strings and mismatched IDs.

Extend `DatasetBundle` with required `source_doc_ids: np.ndarray`. Load only:

    SAFE_CATEGORIES = (
        "comp.graphics",
        "rec.sport.baseball",
        "sci.space",
    )

Call `fetch_20newsgroups(subset="all", categories=SAFE_CATEGORIES, remove=("headers", "footers", "quotes"), shuffle=True, random_state=42)`. Sanitize each document, discard blank results, retain its pre-filter row number in `source_doc_ids`, and validate at least 500 documents and two labels.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run:

    .venv/bin/python -m pytest tests/test_privacy.py tests/test_preprocessing.py tests/test_dataset.py tests/test_tfidf.py tests/test_validation.py -q

Expected: all selected tests pass with no warnings.

- [ ] **Step 5: Commit the dataset boundary**

    git add src/document_system/privacy.py src/document_system/preprocessing.py src/document_system/dataset.py tests/test_privacy.py tests/test_preprocessing.py tests/test_dataset.py
    git commit -m "feat: enforce privacy-safe document input"

---

### Task 2: NumPy vector cosine and sanitized artifact model

**Files:**
- Modify: `src/document_system/search.py`
- Modify: `src/document_system/artifacts.py`
- Modify: `tests/test_search.py`
- Modify: `tests/test_artifacts.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Preserves: `sparse_dot(left_indices, left_data, right_indices, right_data) -> float`
- Changes: `DocumentSearch(..., snippets, labels, target_names, document_ids)`
- Changes: `save_search_artifacts(directory, vectorizer, matrix, snippets, labels, target_names, document_ids)`
- Produces: metadata keys `snippets`, `document_ids`, and `privacy_policy`

- [ ] **Step 1: Write failing vectorization and artifact tests**

Update search fixtures to pass sanitized snippets and source document IDs. Add a source-ID assertion:

    searcher = DocumentSearch(
        vectorizer=vectorizer,
        matrix=matrix,
        snippets=texts,
        labels=np.array([0, 1], dtype=np.int32),
        target_names=("space", "baseball"),
        document_ids=np.array([42, 99], dtype=np.int64),
    )
    assert searcher.search("shuttle orbit", topk=1)[0].doc_id == 42

Monkeypatch `np.intersect1d` and `np.dot` with recording wrappers and assert both are invoked by `sparse_dot`.

Change artifact tests to assert:

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert "texts" not in metadata
    assert metadata["snippets"] == list(snippets)
    assert metadata["document_ids"] == [42, 99]
    assert metadata["privacy_policy"] == "safe-topic-terms-v1"

Add a legacy metadata test that removes `privacy_policy` and expects a `ValueError` containing `rebuild`.

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

    .venv/bin/python -m pytest tests/test_search.py tests/test_artifacts.py tests/test_cli.py -q

Expected: failures from missing vector calls and the old `texts` artifact schema.

- [ ] **Step 3: Implement NumPy cosine and artifact overwrite schema**

Replace the two-pointer loop with:

    _, left_positions, right_positions = np.intersect1d(
        left_indices,
        right_indices,
        assume_unique=True,
        return_indices=True,
    )
    return float(np.dot(left_data[left_positions], right_data[right_positions]))

Change `DocumentSearch` to validate `snippets` and `document_ids`, return the source document ID, and use sanitized snippets for output. Sort equal scores by public source document ID.

Keep `ARTIFACT_VERSION = 1`. Save `snippets`, `document_ids`, and `privacy_policy="safe-topic-terms-v1"`; do not save `texts`. On load, require all new fields before reconstructing `SearchArtifacts`.

Update CLI construction to pass the new fields without changing command-line arguments or output format.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run:

    .venv/bin/python -m pytest tests/test_search.py tests/test_artifacts.py tests/test_cli.py -q

Expected: all selected tests pass.

- [ ] **Step 5: Commit search and artifact changes**

    git add src/document_system/search.py src/document_system/artifacts.py src/document_system/cli.py tests/test_search.py tests/test_artifacts.py tests/test_cli.py
    git commit -m "feat: persist sanitized search artifacts"

---

### Task 3: Remove direct SciPy usage from classification

**Files:**
- Modify: `src/document_system/classification.py`
- Modify: `tests/test_classification.py`

**Interfaces:**
- Removes: `to_sklearn_csr(matrix: SparseMatrix)`
- Preserves: `train_linear_svm`, `predict_sparse`, `evaluate_classifier`
- Consumes: `SparseMatrix.to_dense_rows(row_ids)`

- [ ] **Step 1: Write failing NumPy-batch classification tests**

Remove `from scipy.sparse import isspmatrix_csr` and the adapter test. Add a recording classifier test that verifies each training and prediction call receives `np.ndarray` and at most `batch_size` rows. Add a source scan assertion:

    source = Path("src/document_system/classification.py").read_text()
    assert "scipy" not in source
    assert "csr_matrix" not in source

Keep reproducibility and metric tests unchanged except for removed adapter imports.

- [ ] **Step 2: Run classification tests and verify RED**

Run:

    .venv/bin/python -m pytest tests/test_classification.py -q

Expected: the new direct-SciPy scan fails against the current adapter.

- [ ] **Step 3: Implement bounded dense batches**

Remove the SciPy import and `to_sklearn_csr`. In training, replace CSR slicing with:

    features = matrix.to_dense_rows(batch_ids)

In prediction, build each batch with:

    row_ids = range(start, end)
    predictions.append(model.predict(matrix.to_dense_rows(row_ids)))

Never call `to_dense_rows` with all corpus row IDs at once.

- [ ] **Step 4: Run classification and pipeline tests**

Run:

    .venv/bin/python -m pytest tests/test_classification.py tests/test_pipeline.py -q

Expected: all selected tests pass and no test imports SciPy.

- [ ] **Step 5: Commit classifier changes**

    git add src/document_system/classification.py tests/test_classification.py
    git commit -m "refactor: classify with numpy batches"

---

### Task 4: Integrate source IDs, reports, and licensing

**Files:**
- Create: `DATASET_LICENSE.md`
- Modify: `src/document_system/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `DatasetBundle.source_doc_ids`
- Consumes: sanitized `texts` as snippets
- Produces: existing report paths with new privacy-safe content
- Preserves: `BuildReport` and CLI output shape

- [ ] **Step 1: Write failing pipeline privacy assertions**

Update the small `DatasetBundle` fixture with `source_doc_ids=np.arange(10)`. After build, assert:

    metadata = json.loads((config.runtime_dir / "metadata.json").read_text())
    assert "texts" not in metadata
    assert metadata["document_ids"] == list(range(10))
    assert all(is_safe_text(text) for text in metadata["snippets"])

Assert every `text_snippet` in search and misclassification reports is safe or empty only when the report list itself is empty.

- [ ] **Step 2: Run the pipeline test and verify RED**

Run:

    .venv/bin/python -m pytest tests/test_pipeline.py -q

Expected: fixture construction or artifact assertions fail until source IDs and new artifact calls are integrated.

- [ ] **Step 3: Integrate source IDs through the pipeline**

Use row positions for `train_test_split`, but pass `bundle.source_doc_ids[test_ids]` to classification evaluation. Construct `DocumentSearch` with sanitized texts as `snippets` and full `source_doc_ids`. Pass the same values to `save_search_artifacts`.

Remove exact assumptions about 18,846 documents and 20 classes. Require at least 500 sanitized documents and at least two target names.

- [ ] **Step 4: Add licensing and update README**

Create `DATASET_LICENSE.md` with:

- dataset name and creator attribution;
- UCI DOI `10.24432/C5C323`;
- CC BY 4.0 attribution statement as recorded by UCI;
- Scikit-learn loader source;
- no raw-text redistribution policy;
- safe-topic-term derivation and privacy boundary.

Update README so every number comes from the rebuilt artifacts. Explain:

- three selected categories and filtered document count;
- sanitized safe-vocabulary data and blank removal;
- cleaned whitespace tokenization;
- no synonym expansion because it adds external lexical assumptions and can make contextually wrong replacements;
- `np.intersect1d` plus `np.dot` cosine;
- NumPy dense classifier batches without direct SciPy;
- artifact overwrite behavior and license reference.

- [ ] **Step 5: Run integration tests and documentation checks**

Run:

    .venv/bin/python -m pytest tests/test_pipeline.py tests/test_cli.py -q
    rg -n "scipy|18,846|20개 카테고리|두 포인터" README.md src tests
    git diff --check

Expected: tests pass; any remaining matches are either removed or confined to historical design documents; diff check is clean.

- [ ] **Step 6: Commit pipeline and documentation**

    git add DATASET_LICENSE.md README.md src/document_system/pipeline.py tests/test_pipeline.py
    git commit -m "docs: document privacy-safe dataset experiment"

---

### Task 5: Rebuild artifacts and verify all mission requirements

**Files:**
- Overwrite: `artifacts/reports/confusion_matrix.png`
- Overwrite: `artifacts/reports/matrix_stats.json`
- Overwrite: `artifacts/reports/metrics.json`
- Overwrite: `artifacts/reports/misclassifications.json`
- Overwrite: `artifacts/reports/search_examples.json`
- Overwrite: `artifacts/reports/stage_example.json`
- Overwrite: `artifacts/reports/tfidf_validation.json`
- Overwrite ignored runtime files under `artifacts/runtime/`

**Interfaces:**
- Consumes: unchanged build CLI
- Produces: rebuilt runtime and reports at existing paths

- [ ] **Step 1: Run the complete automated test suite**

Run:

    .venv/bin/python -m pytest -q

Expected: every test passes with no warnings or collection errors.

- [ ] **Step 2: Rebuild in place**

Run:

    .venv/bin/python main.py build

Expected: exit 0; at least 500 documents, three categories, validation error at or below `1e-6`, Accuracy and macro F1 printed.

- [ ] **Step 3: Validate generated JSON, privacy, and artifact schema**

Run a read-only Python check that:

- parses every report JSON;
- asserts `metrics["category_count"] == 3`;
- asserts `tfidf_validation["passed"] is True`;
- asserts `tfidf_validation["max_absolute_error"] <= 1e-6`;
- asserts runtime metadata contains no `texts` key;
- asserts every runtime snippet is non-empty and `is_safe_text(snippet)`;
- asserts every report snippet is safe;
- asserts `privacy_policy == "safe-topic-terms-v1"`.

Expected: all assertions pass.

- [ ] **Step 4: Run the documented Top-5 query**

Run:

    .venv/bin/python main.py --query "space shuttle orbit" --topk 5

Expected: five results containing score, document ID, category, and sanitized snippet.

- [ ] **Step 5: Reconcile README numbers and inspect the final diff**

Compare README document count, train/test counts, vocabulary size, validation errors, classification metrics, matrix statistics, and search examples with generated JSON. Run:

    rg -n "from scipy|import scipy" src tests
    git diff --check
    git status --short

Expected: no direct SciPy matches, clean diff formatting, and only intended files changed.

- [ ] **Step 6: Commit regenerated evidence**

    git add README.md artifacts/reports
    git commit -m "docs: refresh compliant experiment evidence"
