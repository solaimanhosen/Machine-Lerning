import argparse

import pandas as pd

from convnext_tiny import start_training

CSV_PATH = "fig1_imagenet_noise_sigma_calibration_wo_delta_direct.csv"
SIGMA_COLUMN = "sigma_direct_pld"
LEARNING_RATE = 0.0005


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    if SIGMA_COLUMN not in df.columns:
        raise ValueError(f"Missing required column '{SIGMA_COLUMN}' in {CSV_PATH}")

    sigma_values = df[SIGMA_COLUMN].astype(float).tolist()
    total_runs = len(sigma_values)

    for index, sigma in enumerate(sigma_values, start=1):
        print(f"[{index}/{total_runs}] Running with sigma={sigma}, lr={LEARNING_RATE}")
        args = argparse.Namespace(
            use_differential_privacy=True,
            target_epsilon=None,
            learning_rate=LEARNING_RATE,
            sigma=sigma,
            out="direct"
        )
        start_training(args)

    print("\nRun completed successfully.\n")


if __name__ == "__main__":
    main()
