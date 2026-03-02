#!/usr/bin/env python3
import argparse
import csv
from typing import List, Dict
import matplotlib.pyplot as plt


def load_results_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_accuracy(rows: List[Dict[str, str]]) -> float:
    total = len(rows)
    correct = sum(int(r["correct"]) for r in rows)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV result files from medqa_hf_eval.py",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional labels for each model (default: use model_name column).",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="accuracy_bar.png",
        help="Output filename for the accuracy bar chart.",
    )
    args = parser.parse_args()

    accuracies = []
    model_labels = []

    for i, csv_path in enumerate(args.csv_files):
        rows = load_results_csv(csv_path)
        acc = compute_accuracy(rows)

        # Use provided label if given; otherwise infer from model_name in CSV
        if args.labels is not None and i < len(args.labels):
            label = args.labels[i]
        else:
            # Try to infer model_name from first row
            if rows and "model_name" in rows[0]:
                label = rows[0]["model_name"]
            else:
                label = f"model_{i+1}"

        accuracies.append(acc)
        model_labels.append(label)
        print(f"{label}: accuracy = {acc:.3f} (from {len(rows)} examples)")

    # --- Plot bar chart ---
    plt.figure()
    x = range(len(accuracies))
    plt.bar(x, accuracies)
    plt.xticks(x, model_labels, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("MedQA accuracy per model")

    plt.tight_layout()
    plt.savefig(args.output_png, dpi=300)
    print(f"Saved accuracy bar chart to: {args.output_png}")


if __name__ == "__main__":
    main()
