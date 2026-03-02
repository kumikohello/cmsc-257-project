#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_ece(confidences, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1

    ece = 0.0
    bin_acc = []
    bin_conf = []
    N = len(confidences)

    for b in range(n_bins):
        mask = bin_ids == b
        count = np.sum(mask)
        if count == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            continue

        acc = labels[mask].mean()
        conf = confidences[mask].mean()

        bin_acc.append(acc)
        bin_conf.append(conf)

        ece += (count / N) * abs(acc - conf)

    return ece, bins, np.array(bin_acc), np.array(bin_conf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_path", default="ece_plot.png")
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    cols = set(df.columns)

    # Figure out which prob columns we have
    noncot_prob_cols = ["prob_A", "prob_B", "prob_C", "prob_D", "prob_E"]
    cot_prob_cols = ["p_A", "p_B", "p_C", "p_D", "p_E"]

    if all(c in cols for c in noncot_prob_cols):
        prob_cols = noncot_prob_cols
        print("Detected non-CoT CSV (prob_A..prob_E).")
    elif all(c in cols for c in cot_prob_cols):
        prob_cols = cot_prob_cols
        print("Detected CoT/Modal CSV (p_A..p_E).")
    else:
        raise ValueError(
            f"Could not find probability columns in {args.csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Figure out which correctness column we have
    if "correct" in cols:
        label_col = "correct"
    elif "correct_det" in cols:
        label_col = "correct_det"
    else:
        raise ValueError(
            f"Could not find 'correct' or 'correct_det' column in {args.csv_path}."
        )

    # Confidence = max predicted prob over options
    confidences = df[prob_cols].max(axis=1).to_numpy()
    labels = df[label_col].to_numpy().astype(int)

    # Drop NaNs if necessary
    mask = ~np.isnan(confidences)
    confidences = confidences[mask]
    labels = labels[mask]

    ece, bins, bin_acc, bin_conf = compute_ece(confidences, labels, args.n_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(6, 4))
    plt.bar(
        bin_centers,
        bin_acc,
        width=1 / args.n_bins,
        edgecolor="black",
        alpha=0.8,
        label="Empirical accuracy",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect calibration",
    )
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Expected Calibration Error (ECE)\nECE = {ece:.3f}")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=200)

    print(f"ECE = {ece:.4f}")
    print(f"Saved ECE bar chart to: {args.out_path}")


if __name__ == "__main__":
    main()
