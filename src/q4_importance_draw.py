import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get the project root directory
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Read the data
df = pd.read_csv(
    project_root
    / "src"
    / "output"
    / "question4_res"
    / "feature_importance_static_only.csv"
)

# Sort by average importance for better visualization
df["avg_importance"] = (df["importance_judge"] + df["importance_audience"]) / 2
df = df.sort_values("avg_importance", ascending=True)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Y positions for the bars
y_pos = np.arange(len(df))
bar_height = 0.35

# Create horizontal bars
bars1 = ax.barh(
    y_pos - bar_height / 2,
    df["importance_judge"],
    bar_height,
    label="Judge",
    color="red",
    alpha=0.8,
)
bars2 = ax.barh(
    y_pos + bar_height / 2,
    df["importance_audience"],
    bar_height,
    label="Fan Votes",
    color="gold",
    alpha=0.8,
)

# Customize the plot
ax.set_yticks(y_pos)
ax.set_yticklabels(df["feature"])
ax.set_xlabel("Feature Importance")
ax.set_ylabel("Features")
ax.set_title("Feature Importance Comparison: Judge vs Fan")
ax.legend(loc="lower right")

# Add grid for better readability
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = (
    project_root
    / "src"
    / "output"
    / "question4_res"
    / "feature_importance_comparison.png"
)
plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)
print(f"Figure saved to {output_path}")

plt.show()
