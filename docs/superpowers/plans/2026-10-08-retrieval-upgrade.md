# Retrieval upgrade implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement task by task, with failing tests before production changes.

**Goal:** Complete the supplied NLP mission and offline six-step CLI demonstration.
**Architecture:** Extend existing count/CSR and persistence. Keep classification/train and full-corpus/demo spaces separate. Use one common relevance/latency experiment report.
**Tech Stack:** Python >=3.10, NumPy, standard library, existing sklearn/matplotlib/pytest/Ruff/uv.
**Spec:** docs/superpowers/specs/2026-10-08-retrieval-upgrade-design.md

## Global constraints

Preserve loader, redaction, categories, IDs, license, tests, uv.lock, Makefile and CI. Direct algorithms only. Current branch. No dense corpus, external retrieval library, service or frontend. Record only measured results.

## Review focus

Corrupt/mixed artifacts must reject before search. OOV/blank and invalid CLI options must be clear errors. Zero-score held-out queries count in evaluation. Ties use external IDs. Demo evidence must belong to its runtime corpus.

### Task 1: TF variants

Files: tfidf.py, validation.py, tests/test_tf_variants.py. Interface: mode selection, public weight_counts(counts), unchanged default raw.
- [x] Add raw/log sklearn, hand double, empty/OOV/normalization tests; run and confirm failure.
- [x] Implement sparse TF weighting and mode-aware sklearn validation; test green.

### Task 2: retrieval algorithms

Files: bm25.py, inverted_index.py, search.py; tests/test_bm25.py, tests/test_inverted_index.py. Interfaces: shared SearchResult; BM25 from texts or shared counts; scan/inverted strategy.
- [x] Write formula/length/saturation/parameter/tie tests and posting/search equivalence tests; confirm failures.
- [x] Implement sorted postings, candidate accumulation and bounded heap; verify algorithms and regression tests.

### Task 3: artifacts/build

Files: artifacts.py, pipeline.py; new bundle tests. Interface: optional counts on legacy save; load full retrieval bundle without fit.
- [x] Write all-mode roundtrip, corruption and configuration restoration tests; confirm failures.
- [x] Save shared structures and variant weights, digest/schema; integrate build; verify roundtrip and existing pipeline.

### Task 4: reproducible experiments

Files: experiments/tf_variants.py, retrieval_benchmark.py, reporting.py, metrics.py, ablation.py. Interface: run_benchmark(bundle, runtime_dir, reports_dir, query_count=100, repeats=5).
- [x] Write fixture split/leakage/schema/metric/statistics/determinism tests; confirm failures.
- [x] Implement E1-E6, per-class/errors/plots, metadata and report writers. Verify fixture experiment.

### Task 5: presentation CLI

Files: cli.py, presentation.py, main.py, requirements.txt, Makefile, tests/test_demo_cli.py.
- [x] Write fixture demo/compare/menu/error/offline/subprocess tests; confirm failures.
- [x] Implement dispatch and readable tables, offline demo and report consistency checks. Verify installed legacy CLI and new wrapper.

### Task 6: actual experiments/documentation/final review

Files: README.md, docs/experiment-results.md, artifacts/reports/*.
- [x] Run real build and benchmark on all 20 categories; capture real demo output.
- [x] Preserve historical analysis in docs; README quick start plus required implementation/verification/limitations sections.
- [x] Run make check/test/smoke/build; review changes, resolve findings, record limits.

## Execution record

Native execution; no per-task commits required by user. User explicitly authorizes current-branch implementation and routine choices, so redundant approval/worktree handoffs are omitted. Reference repository read for common-result and fair-evaluation ideas only; no implementation copied.


## Verified execution evidence

- Baseline: 83 tests passed before changes. Final: 154 tests passed on Python 3.13.7 and 3.10.22. Ruff/static/document checks, smoke and wheel/sdist build passed.
- Algorithms and persistence followed RED→GREEN; independent review found persisted-posting restore, E5 tie alignment and nested report validation gaps. Reproductions failed first; fixes passed; reviewer verified the three fixes and malformed metadata handling with 28 tests.
- Full build retained 18,309 documents and all 20 categories. Stratified train/test: 14,647/3,662. Benchmarks used 100 shared held-out queries, 5 repetitions, 3 short speed queries at 1k/5k/10k/all.
- Ruling: E5 historical default ties remain internal rows; new E5 uses external IDs to match E2/E3. The full experiment was already running during that fix. Quality was independently recomputed with public-ID ties afterward; on this corpus both E5 numbers were unchanged. Classifier/timing measurements were preserved, and this provenance is recorded in retrieval_benchmark.json. Per-query E2/E3 details were recomputed on exactly the recorded split/query IDs and matched the measured aggregate metrics within 1e-12.
- Ruling: keep working changes on current main branch, as the user explicitly requested. No push, merge, destructive cleanup, or redundant integration question.
- Environment: make absent; downloaded/extracted to /tmp. Network-restricted package build initially failed to resolve setuptools; after authorized dependency download, final make build passed using cached offline dependencies. No project dependency/uv.lock changes.
- Runtime/search fingerprint and index SHA bind benchmark to the corpus and configuration; malformed/incomplete reports cannot display a false PASS.
- Actual demo transcript: docs/demo-output.txt. Detailed final report: docs/development-report.md.

- Follow-up request: organize completed changes into eight feature-focused commits on the current branch; leave push/PR untouched.
