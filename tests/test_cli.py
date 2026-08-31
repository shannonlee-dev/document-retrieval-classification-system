import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import document_system.pipeline as pipeline_module
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


def test_cli_build_prints_dual_tfidf_metrics(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "build_project",
        lambda _config: SimpleNamespace(
            document_count=20,
            classification_vocabulary_size=101,
            search_vocabulary_size=109,
            classification_max_absolute_error=1e-8,
            search_max_absolute_error=2e-8,
            accuracy=0.75,
            macro_f1=0.7,
        ),
    )

    exit_code = main(["build"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "classification_vocabulary=101" in output
    assert "search_vocabulary=109" in output
    assert "classification_max_error=1.000e-08" in output
    assert "search_max_error=2.000e-08" in output


def test_cli_distinguishes_blank_and_oov_queries(tmp_path: Path, capsys) -> None:
    write_small_artifacts(tmp_path)

    blank_exit_code = main(
        ["--artifacts", str(tmp_path), "--query", "", "--topk", "1"]
    )
    blank_error = capsys.readouterr().err
    oov_exit_code = main(
        ["--artifacts", str(tmp_path), "--query", "zzzzunknown", "--topk", "1"]
    )
    oov_error = capsys.readouterr().err

    assert blank_exit_code == oov_exit_code == 2
    assert "blank" in blank_error
    assert "vocabulary" in oov_error


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
