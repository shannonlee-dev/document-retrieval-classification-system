# Module Responsibility Refactor Design

## Goal

Split build reporting, classification, and stop-word ablation into focused
modules without preserving the old combined import paths. Runtime behavior,
generated report schemas, filenames currently present in the working tree, and
CLI behavior remain unchanged.

## Current Problems

- `pipeline.py` coordinates the build while also constructing and writing every
  JSON report and the confusion-matrix image.
- `classification.py` owns model training, prediction, evaluation, and
  visualization.
- `ablation.py` owns retrieval metrics, experiment execution and aggregation,
  and report persistence.

These responsibilities change for different reasons and can be tested through
smaller public modules.

## Package Structure

```text
src/document_system/
├── classification/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── model.py
│   └── visualization.py
├── experiments/
│   ├── __init__.py
│   ├── ablation.py
│   ├── metrics.py
│   └── reporting.py
├── reporting.py
├── pipeline.py
└── ...existing focused modules
```

The old `classification.py` and `ablation.py` modules are removed. Their prior
combined APIs are not re-exported from package `__init__.py` files.

## Responsibilities and Interfaces

### Classification

`classification/model.py` owns:

- `train_linear_svm(...) -> SGDClassifier`
- `predict_sparse(...) -> np.ndarray`
- model-training constants

`classification/evaluation.py` owns:

- `ClassificationReport`
- `evaluate_classifier(...) -> ClassificationReport`
- the model description used in serialized metrics

`classification/visualization.py` owns:

- `save_confusion_matrix(...) -> None`
- plot-specific constants and Matplotlib setup

Callers import from the defining module. The classification package
`__init__.py` remains empty apart from its package docstring.

### Experiments

`experiments/metrics.py` owns:

- `RetrievalMetrics`
- `evaluate_ranked_labels(...) -> RetrievalMetrics`
- `evaluate_label_retrieval(...) -> RetrievalMetrics`

`experiments/ablation.py` owns:

- `AblationVariant`
- `summarize_variants(...) -> dict[str, object]`
- `run_stop_word_ablation(...) -> dict[str, object]`
- the paired-seed stop-word experiment implementation and constants

`experiments/reporting.py` owns:

- `write_stop_word_ablation_report(...) -> dict[str, object]`

The writer calls the experiment runner and persists its returned JSON without
changing the report shape.

### Build Reporting

Top-level `reporting.py` owns build report construction and persistence:

- `write_build_reports(...) -> None`
- JSON serialization helpers
- the maximum number of reported misclassifications

It writes the working tree's current report filenames:

- `dataset_sanitization_report.json`
- `tfidf_sklearn_validation.json`
- `search_index_statistics.json`
- `classification_metrics.json`
- `tfidf_transformation_example.json`
- `classification_error_examples.json`
- `search_result_examples.json`
- `classification_confusion_matrix.png`

`pipeline.py` retains `BuildConfig`, `BuildReport`, dataset loading and
validation, stage coordination, runtime search-artifact persistence, and final
report construction. It delegates build report output to `write_build_reports`.

## Dependency Direction

```text
cli
 └─ pipeline
     ├─ build_stages
     │   ├─ classification.model
     │   ├─ classification.evaluation
     │   ├─ search
     │   └─ validation
     ├─ reporting
     │   └─ classification.visualization
     └─ artifacts

experiments.reporting
 └─ experiments.ablation
     ├─ experiments.metrics
     ├─ classification.model
     └─ classification.evaluation
```

No lower-level numerical, search, or classification module imports a workflow
or reporting module. No compatibility module imports are retained.

## Compatibility and Working-Tree Changes

The old Python import paths are intentionally breaking:

- `document_system.classification` becomes imports from
  `document_system.classification.model`, `.evaluation`, or `.visualization`.
- `document_system.ablation` becomes imports from
  `document_system.experiments.ablation`, `.metrics`, or `.reporting`.

All repository call sites and tests move to the new paths. Historical plan and
spec documents remain unchanged. Existing uncommitted filename changes in
`artifacts.py`, `pipeline.py`, `tests/test_artifacts.py`, and
`tests/test_pipeline.py` are preserved and incorporated rather than reverted.

## Error Handling and Output Contracts

The refactor does not change validation rules, exceptions, metric formulas,
model parameters, generated JSON content, or CLI exit behavior. File writes
continue to create parent directories where the current implementation does.
Build reporting receives already-computed stage results and does not rerun
training, validation, or search.

## Testing

Tests first establish the new import contracts and fail while the new modules
do not exist. Existing behavioral tests are then updated to import each symbol
from its defining module.

Verification covers:

1. classification training, evaluation, and image output tests;
2. ablation metric, experiment, and report output tests;
3. pipeline and artifact integration tests, including the current renamed
   output files;
4. the complete pytest suite and Ruff checks for source and tests;
5. a repository search confirming no executable code uses the removed module
   paths.
