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

TF_MODES = ("raw", "log", "double")
SCORE_TIE_DECIMALS = 12
SEARCH_SCORE_TOLERANCE = 1e-12
MILLISECONDS_PER_SECOND = 1000
MAX_REPORTED_MISCLASSIFICATIONS = 20

DEFAULT_SEARCH_QUERIES = (
    "space shuttle orbit",
    "baseball pitcher season",
    "computer graphics image",
)
