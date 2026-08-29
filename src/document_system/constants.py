"""Shared, stable defaults for the document system."""

from pathlib import Path

DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 6
DEFAULT_RANDOM_STATE = 42
DEFAULT_TOP_K = 5
DEFAULT_TEST_SIZE = 0.2
MINIMUM_DOCUMENTS = 500
SNIPPET_LIMIT = 240

DEFAULT_RUNTIME_DIR = Path("artifacts/runtime")
DEFAULT_REPORTS_DIR = Path("artifacts/reports")
