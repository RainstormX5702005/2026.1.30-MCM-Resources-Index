import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main() -> None:
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 2, 2, 2, 1, 1, 2]
    labels = ["Class 0", "Class 1", "Class 2"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="bilinear")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Counts", rotation=45, labelpad=15)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix Heatmap")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ACU = np.trace(cm) / np.sum(cm)
    txt = f"Overall Accuracy (ACU): {ACU:.4%}"
    ax.annotate(
        text=txt,
        xy=(0.5, -0.1),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
