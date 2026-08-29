"""Run the reproducible default stop-word ablation."""

from pathlib import Path

from document_system.ablation import write_stop_word_ablation_report

REPORT_PATH = Path("artifacts/reports/stop_word_ablation.json")


def main() -> None:
    report = write_stop_word_ablation_report(REPORT_PATH)
    for variant in report["variants"]:
        print(variant)
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
