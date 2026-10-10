import builtins
import json
import os
import subprocess
import sys

import pytest

from document_system.cli import main
from document_system.experiments.retrieval_benchmark import run_benchmark
from document_system.pipeline import BuildConfig, build_from_dataset
from tests.test_retrieval_benchmark import _fixture_bundle


@pytest.fixture
def prepared(tmp_path):
    bundle = _fixture_bundle()
    runtime, reports = tmp_path / "runtime", tmp_path / "reports"
    build_from_dataset(
        bundle,
        BuildConfig(
            runtime_dir=runtime,
            reports_dir=reports,
            batch_size=8,
            epochs=1,
            search_queries=("space orbit",),
        ),
    )
    run_benchmark(
        bundle,
        runtime,
        reports,
        query_count=4,
        repeats=2,
        epochs=1,
        batch_size=8,
        sizes=(10,),
        queries=("space orbit", "baseball game", "rocket"),
    )
    return runtime, reports


@pytest.mark.smoke
def test_demo_runs_all_six_steps_offline_without_training_imports(prepared):
    runtime, reports = prepared
    environment = os.environ.copy()
    environment["NO_COLOR"] = "1"
    environment["MPLCONFIGDIR"] = str(runtime / "not-a-directory")
    (runtime / "not-a-directory").write_text("block plotting")
    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "demo",
            "--artifacts",
            str(runtime),
            "--reports",
            str(reports),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for number in range(1, 7):
        assert f"[{number}/6]" in completed.stdout
    assert "Demo Completed" in completed.stdout
    assert "PASS" in completed.stdout
    assert "Documents     : 40" in completed.stdout
    assert "\x1b" not in completed.stdout
    assert completed.stderr == ""


def test_compare_and_variant_search_use_prepared_indexes(prepared, capsys):
    runtime, reports = prepared
    assert (
        main(
            [
                "compare",
                "--artifacts",
                str(runtime),
                "--reports",
                str(reports),
                "--query",
                "space orbit",
                "--topk",
                "3",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "BM25" in output and "Double" in output and "Top-5" in output
    for method in ["tfidf", "bm25"]:
        assert (
            main(
                [
                    "search",
                    "--artifacts",
                    str(runtime),
                    "--query",
                    "space",
                    "--method",
                    method,
                    "--tf-mode",
                    "log",
                    "--strategy",
                    "inverted",
                    "--topk",
                    "2",
                ]
            )
            == 0
        )
    assert "Search Latency" in capsys.readouterr().out


def test_interactive_recovers_bad_choice_blank_query_and_exits(
    prepared, monkeypatch, capsys
):
    runtime, reports = prepared
    responses = iter(["wrong", "1", "", "1", "space", "6", "q"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(responses))
    assert (
        main(["interactive", "--artifacts", str(runtime), "--reports", str(reports)])
        == 0
    )
    output = capsys.readouterr().out
    assert "Select" in output and "blank" in output and "Demo Completed" in output


@pytest.mark.parametrize("command", ["demo", "compare", "interactive"])
def test_missing_artifacts_are_actionable(tmp_path, capsys, command):
    args = [command, "--artifacts", str(tmp_path)]
    if command == "compare":
        args += ["--query", "space"]
    assert main(args) == 2
    error = capsys.readouterr().err
    assert "python main.py build" in error
    assert "Traceback" not in error


def test_missing_or_mismatched_benchmark_is_not_a_demo_pass(prepared, capsys):
    runtime, reports = prepared
    report_path = reports / "retrieval_benchmark.json"
    payload = json.loads(report_path.read_text())
    payload["metadata"]["corpus_fingerprint"] = "wrong"
    report_path.write_text(json.dumps(payload))
    assert main(["demo", "--artifacts", str(runtime), "--reports", str(reports)]) == 2
    assert "benchmark" in capsys.readouterr().err.lower()
    report_path.unlink()
    assert main(["demo", "--artifacts", str(runtime), "--reports", str(reports)]) == 2
    assert "python main.py benchmark" in capsys.readouterr().err


def test_help_invalid_options_and_ctrl_c(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    for flag, value in [
        ("--method", "dense"),
        ("--tf-mode", "bad"),
        ("--strategy", "bad"),
    ]:
        with pytest.raises(SystemExit) as exit_info:
            main(["search", "--query", "space", flag, value])
        assert exit_info.value.code == 2
    monkeypatch.setattr(
        "document_system.cli._run_search",
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert main(["--query", "space"]) == 130
    assert "cancelled" in capsys.readouterr().err.lower()


def test_raw_inverted_cli_never_rebuilds_persisted_postings(prepared, monkeypatch):
    from document_system.inverted_index import InvertedIndex

    runtime, _ = prepared
    monkeypatch.setattr(
        InvertedIndex, "from_matrix", lambda *_: pytest.fail("postings rebuilt")
    )
    assert (
        main(
            [
                "search",
                "--artifacts",
                str(runtime),
                "--query",
                "space",
                "--strategy",
                "inverted",
                "--topk",
                "2",
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing_accuracy", "invalid_validation", "invalid_latency", "missing_errors"],
)
def test_incomplete_benchmark_is_actionable_without_traceback(
    prepared, capsys, mutation
):
    runtime, reports = prepared
    path = reports / "retrieval_benchmark.json"
    payload = json.loads(path.read_text())
    if mutation == "missing_accuracy":
        payload["tf_variants"][0].pop("accuracy")
    elif mutation == "invalid_validation":
        payload["validation"]["raw"]["passed"] = "yes"
    elif mutation == "invalid_latency":
        payload["inverted_index"][0]["scan_mean_ms"] = "fast"
    else:
        payload["tf_variants"][0].pop("misclassifications")
    path.write_text(json.dumps(payload))
    assert main(["demo", "--artifacts", str(runtime), "--reports", str(reports)]) == 2
    error = capsys.readouterr().err
    assert "python main.py benchmark" in error
    assert "Traceback" not in error


def test_artifact_non_object_json_is_actionable(tmp_path, capsys):
    (tmp_path / "search_index_arrays.npz").write_bytes(b"unused")
    (tmp_path / "search_index_data.json").write_text("[]")
    assert main(["--artifacts", str(tmp_path), "--query", "space"]) == 2
    assert "rebuild" in capsys.readouterr().err.lower()


def test_demo_rejects_benchmark_from_another_index_configuration(prepared, capsys):
    runtime, reports = prepared
    path = reports / "retrieval_benchmark.json"
    payload = json.loads(path.read_text())
    payload["metadata"]["runtime_artifact_sha256"] = "old index configuration"
    path.write_text(json.dumps(payload))
    assert main(["demo", "--artifacts", str(runtime), "--reports", str(reports)]) == 2
    assert "python main.py benchmark" in capsys.readouterr().err


def test_benchmark_cli_dispatches_real_fixture_experiment(
    prepared, tmp_path, monkeypatch, capsys
):
    import document_system.dataset as dataset_module

    runtime, _ = prepared
    reports = tmp_path / "cli_benchmark_reports"
    monkeypatch.setattr(dataset_module, "load_20newsgroups", _fixture_bundle)
    assert (
        main(
            [
                "benchmark",
                "--artifacts",
                str(runtime),
                "--reports",
                str(reports),
                "--queries",
                "4",
                "--repeats",
                "1",
                "--epochs",
                "1",
                "--batch-size",
                "8",
            ]
        )
        == 0
    )
    report = json.loads((reports / "retrieval_benchmark.json").read_text())
    assert report["metadata"]["query_count"] == 4
    assert report["metadata"]["repeats"] == 1
    assert "Benchmark complete" in capsys.readouterr().out
