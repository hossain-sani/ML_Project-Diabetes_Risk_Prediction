"""Manual entry point for selecting, running, and documenting model experiments."""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from src.project_paths import PROCESSED_CSV_PATH, RESULTS_DIR
from src.train_and_evaluate import load_data, prepare_data, run_experiments


MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gaussian Naive Bayes",
    "Linear SVM",
    "Neural Network",
    "LightGBM",
    "CatBoost",
    "XGBoost",
    "Rbf SVM",
    "Polynomial SVM",
]


def choose_models(requested_models):
    if requested_models:
        selected = [name.strip() for name in requested_models.split(",") if name.strip()]
    else:
        print("\nAvailable models:")
        for index, name in enumerate(MODEL_NAMES, start=1):
            print(f"  {index}. {name}")
        answer = input(
            "\nEnter model numbers separated by commas, or press Enter for all models: "
        ).strip()
        if not answer:
            selected = MODEL_NAMES.copy()
        else:
            try:
                selected = [MODEL_NAMES[int(value) - 1] for value in answer.split(",")]
            except (ValueError, IndexError) as error:
                raise ValueError("Choose valid model numbers separated by commas.") from error

    if not selected:
        raise ValueError("Select at least one model.")
    if any(name not in MODEL_NAMES for name in selected):
        raise ValueError("Use the exact model names listed by --list-models.")
    return list(dict.fromkeys(selected))


def write_doc_report(results, selected_models, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report:
        report.write("TYPE 2 DIABETES RISK PREDICTION - MODEL REPORT\n")
        report.write("=" * 60 + "\n\n")
        report.write("Selected models:\n")
        report.write("\n".join(f"- {name}" for name in selected_models))
        report.write("\n\nResults:\n")
        report.write(results.to_string(index=False))
        report.write("\n\nNote: This is a research/educational comparison, not a medical diagnosis.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Select models manually and save CSV, ROC, and DOC outputs."
    )
    parser.add_argument(
        "--models",
        help='Comma-separated names, e.g. "Logistic Regression,Random Forest"',
    )
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--data", default=PROCESSED_CSV_PATH, type=Path)
    args = parser.parse_args()

    if args.list_models:
        print("\n".join(MODEL_NAMES))
        return

    selected_models = choose_models(args.models)
    include_kernel_svm = any(name in selected_models for name in ("Rbf SVM", "Polynomial SVM"))
    data_frame = load_data(args.data)
    prepared_data = prepare_data(data_frame)
    results, _ = run_experiments(
        prepared_data,
        run_kernel_svm=include_kernel_svm,
        selected_models=selected_models,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "model_report.doc"
    write_doc_report(results, selected_models, report_path)
    print(f"\nSaved report to {report_path}")
    print("The CSV comparison and ROC chart are saved in the results folder.")


if __name__ == "__main__":
    main()
