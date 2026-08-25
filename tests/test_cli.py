import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from document_system.artifacts import save_search_artifacts
from document_system.cli import main
from document_system.preprocessing import EnglishPreprocessor
from document_system.tfidf import NumpyTfidfVectorizer


def write_small_artifacts(path: Path) -> None:
    snippets = ["space shuttle orbit", "baseball pitcher game"]
    document_ids = np.array([42, 99], dtype=np.int64)
    labels = np.array([0, 1], dtype=np.int32)
    vectorizer = NumpyTfidfVectorizer(
        EnglishPreprocessor(stop_words=frozenset())
    )
    matrix = vectorizer.fit_transform(snippets)
    save_search_artifacts(
        path, vectorizer, matrix, snippets, labels, ("space", "baseball"), document_ids
    )


def test_cli_search_prints_rank_score_id_and_snippet(
    tmp_path: Path, capsys
) -> None:
    write_small_artifacts(tmp_path)

    exit_code = main(
        ["--artifacts", str(tmp_path), "--query", "space orbit", "--topk", "1"]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "1." in output
    assert "score=" in output
    assert "doc_id=" in output
    assert "category=space" in output


def test_cli_missing_artifacts_explains_build_command(
    tmp_path: Path, capsys
) -> None:
    exit_code = main(
        ["--artifacts", str(tmp_path), "--query", "space", "--topk", "1"]
    )

    assert exit_code == 2
    assert "python main.py build" in capsys.readouterr().err


def test_search_process_does_not_import_build_only_matplotlib(
    tmp_path: Path,
) -> None:
    write_small_artifacts(tmp_path)
    invalid_config_dir = tmp_path / "matplotlib-config-file"
    invalid_config_dir.write_text("not a directory", encoding="utf-8")
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(invalid_config_dir)
    environment["PYTHONPATH"] = str(Path("src").resolve())

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--artifacts",
            str(tmp_path),
            "--query",
            "space orbit",
            "--topk",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
