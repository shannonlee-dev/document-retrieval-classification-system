"""Persistence for reproducible experiment reports."""

from __future__ import annotations

import json
from pathlib import Path

from .ablation import run_stop_word_ablation


def write_stop_word_ablation_report(path: str | Path) -> dict[str, object]:
    """Run the stop-word ablation and persist its JSON report."""

    report = run_stop_word_ablation()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
