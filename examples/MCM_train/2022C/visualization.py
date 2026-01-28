import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from configs.config import DATA_DIR, OUTPUT_DIR


def main():
    gold_file = OUTPUT_DIR / "PROCESSED_GOLD.csv"
    if not gold_file.exists():
        print(f"File {gold_file} does not exist. Please run feature.py first.")
        return

    df = pd.read_csv(gold_file)
    df["Date"] = pd.to_datetime(df["Date"])

    figure, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=df, x="Date", y="USD (PM)", ax=ax, label="Actual Price", color="red"
    )
    sns.lineplot(
        data=df,
        x="Date",
        y="PredictVal",
        ax=ax,
        label="Predicted Price",
        color="yellow",
    )

    ax.set_title("Gold Price: Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
