import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        "Epoch": [f"Ep{i}" for i in range(1, 11)],
        "Sample_Count": np.random.randint(100, 500, 10),  # 柱状图数据 (量)
        "Accuracy": np.linspace(0.5, 0.95, 10)
        + np.random.normal(0, 0.02, 10),  # 折线图数据 (率)
    }
)


def main(df):

    sns.set_theme(style="white", palette="muted")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 只需要指定 ax=ax1，其他全自动
    sns.barplot(
        data=df, x="Epoch", y="Sample_Count", color="#BF27D4", alpha=0.6, ax=ax1
    )

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_ylabel("Sample Count", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x="Epoch",
        y="Accuracy",
        color="crimson",
        linewidth=4,
        marker="x",
        ax=ax2,
    )

    ax2.set_ylabel("Validation Accuracy", rotation=90, fontsize=12, color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    plt.title("Training Progress: Samples vs Accuracy", fontsize=14, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    main(df)
