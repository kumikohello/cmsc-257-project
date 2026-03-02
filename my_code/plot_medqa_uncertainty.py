#!/usr/bin/env python3
"""
Plot uncertainty diagnostics from MedQA evaluation CSVs.

Each CSV is expected to have columns:
  question_id, gold, pred_det, correct_det, logprob_det,
  n_samples, entropy, p_A, p_B, p_C, p_D, p_E

You can pass one or more CSV files and (optionally) labels for models.

Usage example:

  python3 plot_medqa_uncertainty.py \
      --csv_files gemma_medqa_us_test.csv mistral_medqa_us_test.csv \
      --labels Gemma Mistral \
      --out_dir plots

This will produce:
  - For each model:
      entropy_box_<label>.png
      calibration_<label>.png
      roc_entropy_<label>.png
  - One combined plot:
      accuracy_bar.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ---------------------- Plot helpers ---------------------- #

def plot_entropy_box(df: pd.DataFrame, label: str, out_path: str):
    """Boxplot: entropy vs correctness (0/1)."""
    plt.figure(figsize=(5, 4))

    # Make sure correct_det is int/binary
    df = df.copy()
    df["correct_det"] = df["correct_det"].astype(int)

    # Boxplot by correctness
    df.boxplot(column="entropy", by="correct_det")
    plt.xlabel("Correct (0 = wrong, 1 = correct)")
    plt.ylabel("Predictive entropy")
    plt.title(f"Entropy vs correctness ({label})")
    plt.suptitle("")  # remove default pandas title
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved entropy boxplot to {out_path}")


def plot_calibration(df: pd.DataFrame, label: str, out_path: str, n_bins: int = 10):
    """
    Calibration plot: predicted confidence vs empirical accuracy.

    We define confidence as max over p_A..p_E for each question.
    """
    df = df.copy()

    # Compute predicted confidence as max option probability
    prob_cols = ["p_A", "p_B", "p_C", "p_D", "p_E"]
    df["p_pred"] = df[prob_cols].max(axis=1)

    # Bin by confidence
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["conf_bin"] = pd.cut(df["p_pred"], bins=bins, include_lowest=True)

    grouped = df.groupby("conf_bin").agg(
        mean_conf=("p_pred", "mean"),
        acc=("correct_det", "mean"),
        count=("correct_det", "size"),
    ).reset_index()

    # Drop bins with no data
    grouped = grouped.dropna(subset=["mean_conf", "acc"])

    plt.figure(figsize=(5, 4))
    plt.plot(grouped["mean_conf"], grouped["acc"], marker="o", label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    plt.xlabel("Predicted confidence (max p(option))")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Calibration plot ({label})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved calibration plot to {out_path}")


def plot_roc_entropy(df: pd.DataFrame, label: str, out_path: str):
    """
    ROC curve using entropy to detect mistakes.

    Positive class = error (incorrect answer)
    Score = entropy (higher entropy -> more likely to be incorrect)
    """
    df = df.copy()
    # y_true: 1 for error, 0 for correct
    y_true = 1 - df["correct_det"].astype(int).values
    y_score = df["entropy"].astype(float).values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC (entropy → error detection) ({label})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve to {out_path}")


def compute_accuracy(df: pd.DataFrame) -> float:
    """Compute mean accuracy from correct_det column."""
    return float(df["correct_det"].mean())


def plot_accuracy_bar(model_labels, accuracies, out_path: str):
    """Bar plot of accuracy per model."""
    plt.figure(figsize=(6, 4))
    x = np.arange(len(model_labels))
    plt.bar(x, accuracies)
    plt.xticks(x, model_labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy per model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved accuracy bar plot to {out_path}")


# ---------------------- Main script ---------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV files from medqa_entropy_eval.py",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        help="Optional labels for each CSV (e.g., model names). "
             "If not provided, filenames will be used.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins for calibration plot.",
    )
    args = parser.parse_args()

    csv_files = args.csv_files
    labels = args.labels
    out_dir = args.out_dir
    n_bins = args.bins

    os.makedirs(out_dir, exist_ok=True)

    if labels is None or len(labels) == 0:
        # Use base filenames as labels
        labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    if len(labels) != len(csv_files):
        raise ValueError("Number of labels must match number of csv_files.")

    accuracies = []

    for csv_path, label in zip(csv_files, labels):
        print(f"\n=== Loading {csv_path} ({label}) ===")
        df = pd.read_csv(csv_path)

        # 1) Entropy vs correctness boxplot
        out_entropy = os.path.join(out_dir, f"entropy_box_{label}.png")
        plot_entropy_box(df, label, out_entropy)

        # 2) Calibration plot
        out_calib = os.path.join(out_dir, f"calibration_{label}.png")
        plot_calibration(df, label, out_calib, n_bins=n_bins)

        # 3) ROC curve (entropy)
        out_roc = os.path.join(out_dir, f"roc_entropy_{label}.png")
        plot_roc_entropy(df, label, out_roc)

        # 4) Accuracy summary
        acc = compute_accuracy(df)
        accuracies.append(acc)
        print(f"Accuracy for {label}: {acc:.3f}")

    # 5) If multiple models, accuracy bar plot
    if len(csv_files) > 1:
        out_bar = os.path.join(out_dir, "accuracy_bar.png")
        plot_accuracy_bar(labels, accuracies, out_bar)


if __name__ == "__main__":
    main()
