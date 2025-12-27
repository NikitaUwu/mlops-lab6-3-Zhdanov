import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/dataset.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Приводим таргет к единому имени
    if "churn" not in df.columns:
        if "target" in df.columns:
            df = df.rename(columns={"target": "churn"})
        else:
            raise ValueError("Dataset must contain 'churn' (or 'target' to be renamed).")

    os.makedirs("data/processed", exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["churn"])
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Wrote: {TRAIN_PATH} ({len(train_df)} rows), {TEST_PATH} ({len(test_df)} rows)")

if __name__ == "__main__":
    main()
