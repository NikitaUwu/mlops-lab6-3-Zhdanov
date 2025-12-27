import argparse
import os
import pandas as pd
import numpy as np


RAW_PATH = "data/raw/dataset.csv"


def _is_empty_csv(path: str) -> bool:
    if not os.path.exists(path):
        return True
    try:
        df = pd.read_csv(path)
        return len(df) == 0
    except Exception:
        return True


def generate_synthetic_churn(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"C{100000 + i}" for i in range(n)]
    tenure = rng.integers(0, 73, size=n)  # months: 0..72
    monthly_charges = rng.uniform(20, 120, size=n).round(2)
    total_charges = (monthly_charges * np.maximum(1, tenure) + rng.normal(0, 50, size=n)).clip(0).round(2)

    gender = rng.choice(["M", "F"], size=n)
    senior_citizen = rng.integers(0, 2, size=n)
    partner = rng.choice(["Yes", "No"], size=n, p=[0.45, 0.55])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    # “Скрытая” логика churn для синтетики (чтобы метрики были адекватные)
    # Чем меньше tenure и чем выше monthly_charges, тем выше риск churn
    logit = (
        -2.0
        + 0.03 * (monthly_charges - 60)
        - 0.02 * tenure
        + 0.8 * (partner == "No").astype(float)
        + 0.6 * (dependents == "No").astype(float)
        + 0.4 * senior_citizen.astype(float)
        + rng.normal(0, 0.6, size=n)
    )
    p = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, size=n) < p).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "churn": churn,
        }
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="Number of rows to generate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--force", action="store_true", help="Overwrite even if dataset is not empty")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)

    if not args.force and not _is_empty_csv(RAW_PATH):
        print(f"{RAW_PATH} already exists and is not empty. Use --force to overwrite.")
        return

    df = generate_synthetic_churn(args.n, args.seed)
    df.to_csv(RAW_PATH, index=False)
    print(f"Wrote {RAW_PATH} with {len(df)} rows.")


if __name__ == "__main__":
    main()
