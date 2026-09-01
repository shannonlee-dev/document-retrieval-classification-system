# Module Responsibility Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split build reporting, classification, and stop-word ablation into focused modules and fully migrate repository imports to the new package paths.

**Architecture:** Classification becomes a package with model, evaluation, and visualization modules. Ablation becomes an `experiments` package with metrics, experiment execution, and report persistence modules. The build pipeline delegates generated report output to a top-level `reporting.py` while retaining orchestration and runtime search-index persistence.

**Tech Stack:** Python 3.10+, NumPy, Scikit-learn, Matplotlib, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-01-module-responsibility-refactor-design.md`

## Global Constraints

- Do not preserve compatibility imports for `document_system.classification` or `document_system.ablation`.
- Preserve model behavior, metric formulas, validation rules, exceptions, CLI behavior, and generated report JSON schemas.
- Preserve the current working-tree output filenames, including `search_index_arrays.npz`, `search_index_data.json`, and the renamed report files.
- Preserve all pre-existing user changes; do not revert or overwrite them.
- Add no external dependency.
- Historical plans and specs remain unchanged.
- Because user-owned changes overlap `pipeline.py` and `tests/test_pipeline.py`, do not create implementation commits unless the user explicitly asks after reviewing the final diff.

---

### Task 1: Split classification into model, evaluation, and visualization modules

**Files:**
- Create: `src/document_system/classification/__init__.py`
- Create: `src/document_system/classification/model.py`
- Create: `src/document_system/classification/evaluation.py`
- Create: `src/document_system/classification/visualization.py`
- Delete: `src/document_system/classification.py`
- Modify: `src/document_system/build_stages.py:11`
- Modify: `src/document_system/pipeline.py:11`
- Modify: `tests/test_classification.py`

**Interfaces:**
- Produces: `train_linear_svm` and `predict_sparse` in `classification.model` with their existing signatures and return types.
- Produces: `ClassificationReport` and `evaluate_classifier` in `classification.evaluation` with their existing contracts.
- Produces: `save_confusion_matrix` in `classification.visualization` with its existing contract.
- Consumes: existing `SparseMatrix`, privacy-safe snippets, and classification constants without changing their contracts.

- [ ] **Step 1: Change classification tests to require the new import paths**

Replace the combined imports at the top of `tests/test_classification.py` with:

```python
import document_system.classification.model as model_module
from document_system.classification.evaluation import evaluate_classifier
from document_system.classification.model import predict_sparse, train_linear_svm
from document_system.classification.visualization import save_confusion_matrix
```

Change the bounded-batch monkeypatch to:

```python
monkeypatch.setattr(model_module, "SGDClassifier", RecordingClassifier)
```

Change the no-SciPy source check to inspect all classification implementation files:

```python
def test_classification_has_no_direct_scipy_dependency() -> None:
    package = Path("src/document_system/classification")
    source = "\n".join(path.read_text() for path in package.glob("*.py"))

    assert "scipy" not in source
    assert "csr_matrix" not in source
```

- [ ] **Step 2: Run the classification test to verify RED**

Run: `.venv/bin/pytest tests/test_classification.py -q`

Expected: collection fails because `document_system.classification.model`, `.evaluation`, and `.visualization` do not exist.

- [ ] **Step 3: Create the classification package**

Create `classification/__init__.py` with only:

```python
"""Linear document-classification components."""
```

Move `LINEAR_SVM_LOSS`, `SGD_MAX_ITER_PER_BATCH`, `train_linear_svm`, and
`predict_sparse` from current `classification.py` lines 21, 23, and 53-111
unchanged into `classification/model.py`.

`model.py` imports `numpy`, `SGDClassifier`, classification defaults, and `SparseMatrix`. It contains no evaluation, privacy, Matplotlib, or filesystem imports.

Move `LINEAR_SVM_DESCRIPTION`, `ClassificationReport`, and
`evaluate_classifier` from current `classification.py` lines 22, 33-50, and
114-157 unchanged into `classification/evaluation.py`; import `predict_sparse`
from `.model`.

Move `save_confusion_matrix` and every `CONFUSION_MATRIX_*` constant into `classification/visualization.py`. Configure the `Agg` backend there and import `ClassificationReport` from `.evaluation`.

- [ ] **Step 4: Migrate production classification imports and remove the old module**

Use these imports in `build_stages.py`:

```python
from .classification.evaluation import ClassificationReport, evaluate_classifier
from .classification.model import train_linear_svm
```

Temporarily use this import in `pipeline.py` until Task 3 moves it into build reporting:

```python
from .classification.visualization import save_confusion_matrix
```

Delete `src/document_system/classification.py`; do not create re-exports in `classification/__init__.py`.

- [ ] **Step 5: Run focused classification and stage tests to verify GREEN**

Run: `.venv/bin/pytest tests/test_classification.py tests/test_build_stages.py tests/test_pipeline.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Run focused lint and inspect the diff**

Run: `.venv/bin/ruff check src/document_system/classification src/document_system/build_stages.py src/document_system/pipeline.py tests/test_classification.py`

Expected: exit 0.

Run: `git diff --check -- src/document_system/classification.py src/document_system/classification src/document_system/build_stages.py src/document_system/pipeline.py tests/test_classification.py`

Expected: no whitespace errors.

---

### Task 2: Split experiment metrics, execution, and persistence

**Files:**
- Create: `src/document_system/experiments/__init__.py`
- Create: `src/document_system/experiments/metrics.py`
- Create: `src/document_system/experiments/ablation.py`
- Create: `src/document_system/experiments/reporting.py`
- Delete: `src/document_system/ablation.py`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Consumes: `classification.model`, `classification.evaluation`, `SparseMatrix`, `NumpyTfidfVectorizer`, and dataset/privacy contracts.
- Produces: retrieval metric functions in `experiments.metrics`.
- Produces: ablation aggregation and execution in `experiments.ablation`.
- Produces: `write_stop_word_ablation_report(path) -> dict[str, object]` in `experiments.reporting`.

- [ ] **Step 1: Change ablation tests to require the new modules**

Replace the old imports with:

```python
import json

import pytest

import document_system.experiments.ablation as ablation_module
import document_system.experiments.reporting as reporting_module
from document_system.experiments.ablation import (
    AblationVariant,
    run_stop_word_ablation,
    summarize_variants,
)
from document_system.experiments.metrics import evaluate_ranked_labels
from document_system.experiments.reporting import write_stop_word_ablation_report
```

Add a persistence-boundary test:

```python
def test_write_stop_word_ablation_report_persists_runner_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    expected = {"dataset": "small", "variants": []}
    monkeypatch.setattr(
        reporting_module,
        "run_stop_word_ablation",
        lambda: expected,
    )

    output = tmp_path / "ablation.json"
    result = write_stop_word_ablation_report(output)

    assert result == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected
```

- [ ] **Step 2: Run ablation tests to verify RED**

Run: `.venv/bin/pytest tests/test_ablation.py -q`

Expected: collection fails because `document_system.experiments` does not exist.

- [ ] **Step 3: Create experiment metrics and ablation modules**

Create `experiments/__init__.py` with only:

```python
"""Reproducible document-system experiments."""
```

Move `RETRIEVAL_TOP_K`, `RetrievalMetrics`, `evaluate_ranked_labels`, and
`evaluate_label_retrieval` from current `ablation.py` lines 27, 31-34, and
37-121 unchanged into `experiments/metrics.py`.

Move `ABLATION_SEEDS`, `AblationVariant`, `summarize_variants`, `run_stop_word_ablation`, and `_run_stop_word_ablation_seed` to `experiments/ablation.py`. Replace imports with:

```python
from ..classification.evaluation import evaluate_classifier
from ..classification.model import train_linear_svm
from ..constants import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_TEST_SIZE
from ..dataset import DatasetBundle, load_20newsgroups
from ..preprocessing import DEFAULT_STOP_WORDS, EnglishPreprocessor
from ..privacy import make_safe_snippet
from ..tfidf import NumpyTfidfVectorizer
from .metrics import evaluate_label_retrieval
```

`experiments/ablation.py` contains no JSON or filesystem imports.

- [ ] **Step 4: Create experiment report persistence and remove the old module**

Implement `experiments/reporting.py` with the existing output behavior:

```python
"""Persistence for reproducible experiment reports."""

from __future__ import annotations

import json
from pathlib import Path

from .ablation import run_stop_word_ablation


def write_stop_word_ablation_report(path: str | Path) -> dict[str, object]:
    report = run_stop_word_ablation()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
```

Delete `src/document_system/ablation.py`; do not add compatibility re-exports.

- [ ] **Step 5: Run ablation tests to verify GREEN**

Run: `.venv/bin/pytest tests/test_ablation.py -q`

Expected: all tests pass, including the new persistence test.

- [ ] **Step 6: Run focused lint and inspect the diff**

Run: `.venv/bin/ruff check src/document_system/experiments tests/test_ablation.py`

Expected: exit 0.

Run: `git diff --check -- src/document_system/ablation.py src/document_system/experiments tests/test_ablation.py`

Expected: no whitespace errors.

---

### Task 3: Extract build report persistence from the pipeline

**Files:**
- Create: `src/document_system/reporting.py`
- Create: `tests/test_reporting.py`
- Modify: `src/document_system/pipeline.py:5-205`

**Interfaces:**
- Consumes: `DatasetBundle`, `ClassificationStageResult`, `SearchStageResult`, report directory, epoch count, and random state.
- Produces: `write_build_reports` returning `None`.
- Preserves: current eight report filenames and every serialized payload field.

- [ ] **Step 1: Add a failing build-report output test**

Create `tests/test_reporting.py` with real `DatasetBundle`,
`ClassificationStageResult`, and `SearchStageResult` values, then require the
new public module and call:

```python
write_build_reports(
    tmp_path,
    bundle,
    classification,
    search,
    epochs=2,
    random_state=42,
)
```

Assert the eight exact output filenames, a nonempty PNG, classification metric
fields, and both validation fit scopes. The existing pipeline test remains the
end-to-end contract for report schemas and runtime filenames.

- [ ] **Step 2: Run the pipeline responsibility test to verify RED**

Run: `.venv/bin/pytest tests/test_reporting.py -q`

Expected: collection fails because `document_system.reporting` does not exist.

- [ ] **Step 3: Implement top-level build reporting**

Create `src/document_system/reporting.py` with this public signature:

```python
def write_build_reports(
    reports_dir: Path,
    bundle: DatasetBundle,
    classification: ClassificationStageResult,
    search: SearchStageResult,
    *,
    epochs: int,
    random_state: int,
) -> None:
```

Move `MAX_REPORTED_MISCLASSIFICATIONS`, `_write_json`, and `_validation_payload` from `pipeline.py`. Move current `pipeline.py` lines 94-165, from report-directory creation through confusion-matrix persistence, into this function. Keep these exact filenames:

```python
reports_dir / "dataset_sanitization_report.json"
reports_dir / "tfidf_sklearn_validation.json"
reports_dir / "search_index_statistics.json"
reports_dir / "classification_metrics.json"
reports_dir / "tfidf_transformation_example.json"
reports_dir / "classification_error_examples.json"
reports_dir / "search_result_examples.json"
reports_dir / "classification_confusion_matrix.png"
```

Import `save_confusion_matrix` from `.classification.visualization`. Use `epochs` and `random_state` parameters when constructing classification metrics. Preserve the privacy retained-count validation before writing any report.

- [ ] **Step 4: Reduce pipeline to orchestration**

Remove JSON, `save_confusion_matrix`, `DEFAULT_TEST_SIZE`, `MAX_REPORTED_MISCLASSIFICATIONS`, `ValidationResult`, `_write_json`, and `_validation_payload` from `pipeline.py`.

Import:

```python
from .reporting import write_build_reports
```

After both build stages complete, call:

```python
write_build_reports(
    config.reports_dir,
    bundle,
    classification,
    search,
    epochs=config.epochs,
    random_state=config.random_state,
)
```

Retain the `save_search_artifacts` call and `BuildReport` construction in `pipeline.py`.

- [ ] **Step 5: Run responsibility and integration tests to verify GREEN**

Run: `.venv/bin/pytest tests/test_reporting.py tests/test_pipeline.py tests/test_artifacts.py tests/test_cli.py -q`

Expected: all selected tests pass; report and runtime filenames match the current working-tree contract.

- [ ] **Step 6: Run focused lint and inspect the diff**

Run: `.venv/bin/ruff check src/document_system/reporting.py src/document_system/pipeline.py tests/test_reporting.py tests/test_pipeline.py`

Expected: exit 0.

Run: `git diff --check -- src/document_system/reporting.py src/document_system/pipeline.py tests/test_reporting.py tests/test_pipeline.py`

Expected: no whitespace errors.

---

### Task 4: Verify the complete migration

**Files:**
- Verify: `src/document_system/**/*.py`
- Verify: `tests/**/*.py`
- Verify: current non-historical documentation

**Interfaces:**
- Confirms: removed combined module paths have no executable callers.
- Confirms: full test and lint suites pass with the new package structure.

- [ ] **Step 1: Search for stale executable imports**

Run:

```bash
rg -n 'from document_system\.(classification|ablation) import|import document_system\.(classification|ablation) as|from \.classification import|from \.ablation import' src tests main.py
```

Expected: no matches. Imports containing `.classification.model`, `.classification.evaluation`, `.classification.visualization`, or `.experiments.*` are valid.

- [ ] **Step 2: Confirm removed files are absent and new files exist**

Run:

```bash
test ! -e src/document_system/classification.py
test ! -e src/document_system/ablation.py
test -f src/document_system/classification/model.py
test -f src/document_system/classification/evaluation.py
test -f src/document_system/classification/visualization.py
test -f src/document_system/experiments/metrics.py
test -f src/document_system/experiments/ablation.py
test -f src/document_system/experiments/reporting.py
test -f src/document_system/reporting.py
```

Expected: exit 0.

- [ ] **Step 3: Run the complete test suite**

Run: `.venv/bin/pytest -q`

Expected: all tests pass with zero failures or collection errors.

- [ ] **Step 4: Run Ruff across production and test code**

Run: `.venv/bin/ruff check src tests main.py`

Expected: exit 0.

- [ ] **Step 5: Check formatting and review only task-related changes**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Review the task-related paths while leaving all pre-existing user-owned report, documentation, artifact, and filename changes intact. Do not stage or commit the overlapping implementation diff without a separate user request.
