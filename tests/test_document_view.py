import builtins
from types import SimpleNamespace

import numpy as np
import pytest

import document_system.dataset as dataset_module
from document_system.cli import main
from document_system.pipeline import BuildConfig, build_from_dataset


@pytest.fixture
def documents(tmp_path, monkeypatch, capsys):
    full_text = "space orbit rocket " * 40 + "end of the complete document"
    fake = SimpleNamespace(
        data=["alice@example.com", full_text] + ["baseball pitcher game"] * 499,
        target=np.array([0] + [i // 25 for i in range(500)]),
        target_names=[f"category.{i}" for i in range(20)],
    )
    monkeypatch.setattr(dataset_module, "fetch_20newsgroups", lambda **_: fake)
    bundle = dataset_module.load_20newsgroups()
    runtime = tmp_path / "runtime"
    reports = tmp_path / "reports"
    build_from_dataset(
        bundle,
        BuildConfig(
            runtime_dir=runtime,
            reports_dir=reports,
            epochs=1,
            batch_size=128,
            search_queries=("space",),
        ),
    )
    capsys.readouterr()
    return fake, runtime, reports, full_text


def _run_menu(monkeypatch, runtime, reports, responses):
    answers = iter(responses)

    def input_until_eof(_prompt):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", input_until_eof)
    return main(["interactive", "--artifacts", str(runtime), "--reports", str(reports)])


def test_document_menu_shows_full_text_by_original_id_and_loads_cache_once(
    documents, monkeypatch, capsys
):
    fake, runtime, reports, full_text = documents
    loaded = False

    def offline_fetch(**options):
        nonlocal loaded
        assert options["download_if_missing"] is False
        if loaded:
            raise OSError("document cache was loaded twice")
        loaded = True
        return fake

    monkeypatch.setattr(dataset_module, "fetch_20newsgroups", offline_fetch)
    assert _run_menu(monkeypatch, runtime, reports, ["7", "1", "7", "26", "q"]) == 0
    output = capsys.readouterr().out
    assert " ".join(full_text.split()) in " ".join(output.split())
    assert "Document ID: 1" in output
    assert "Category: category.0" in output
    assert "Document ID: 26" in output
    assert "Category: category.1" in output
    assert "Input error" not in output


@pytest.mark.parametrize("document_id", ["", "abc", "-1", "0", "501"])
def test_document_menu_recovers_invalid_ids_without_loading_data(
    documents, monkeypatch, capsys, document_id
):
    _, runtime, reports, _ = documents

    def unavailable_cache(**_options):
        pytest.fail("invalid document IDs must not load the dataset")

    monkeypatch.setattr(dataset_module, "fetch_20newsgroups", unavailable_cache)
    assert _run_menu(monkeypatch, runtime, reports, ["7", document_id, "q"]) == 0
    assert "Input error" in capsys.readouterr().out


def test_document_menu_recovers_missing_cache(documents, monkeypatch, capsys):
    _, runtime, reports, _ = documents

    def unavailable_cache(**options):
        assert options["download_if_missing"] is False
        raise OSError("20Newsgroups dataset not found")

    monkeypatch.setattr(dataset_module, "fetch_20newsgroups", unavailable_cache)
    assert _run_menu(monkeypatch, runtime, reports, ["7", "1", "1", "space", "q"]) == 0
    output = capsys.readouterr().out
    assert "cache" in output.lower()
    assert "Input error" in output
    assert "Search Latency" in output


def test_document_menu_rejects_different_corpus(documents, monkeypatch, capsys):
    fake, runtime, reports, _ = documents
    fake.data[1] = "different corpus document"
    assert _run_menu(monkeypatch, runtime, reports, ["7", "1", "q"]) == 0
    output = capsys.readouterr().out
    assert "mismatch" in output.lower()
    assert "different corpus document" not in output


def test_document_menu_exits_on_eof_at_document_prompt(documents, monkeypatch):
    _, runtime, reports, _ = documents
    answers = iter(["7"])
    prompts = []

    def input_until_eof(_prompt):
        prompts.append(_prompt)
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", input_until_eof)
    assert (
        main(["interactive", "--artifacts", str(runtime), "--reports", str(reports)])
        == 0
    )
    assert prompts[-1] == "Document ID: "
