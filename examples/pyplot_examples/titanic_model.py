import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.manifold import TSNE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance

from typing import Optional

from configs.encode import set_encoding_method, FIG_DIR


"""本次示例更为全面的展示了特征工程中的以下内容：

1. 如何处理缺失的离散值与连续值 ———— 离散值特殊填补，而连续值可以采用 KNN 或 MICE 插补，对于特殊的时间序列可以用线性插值法
2. 进行降维的可视化分析 ———— 使用 t-SNE 进行降维后的可视化，观测数据处理的准确程度

此外，本示例更多展示的是 HistGradientBoostingClassifier 的使用以及其准确性的校验。

为了检测模型的泛化能力，本代码又加入了混淆矩阵的计算与绘制过程，利用混淆矩阵热力气泡图分析模型的准确性。

在阅读本例之前，建议先学习 Iris 数据集的可视化示例。
"""

set_encoding_method()


def plot_tsne(X, titanic_df, fig_path=None, perplexity=50, random_state=42):
    """使用 t-SNE 对数据集进行分析并进行其散点图绘制"""
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    X_embedded = tsne.fit_transform(X)
    tsne_df = pd.DataFrame(
        X_embedded, columns=["TSNE-1", "TSNE-2"], index=titanic_df.index
    )

    tsne_df["Cabin"] = titanic_df["Cabin"].astype(str)
    tsne_df["Survived_lbl"] = titanic_df["Survived"].map(
        {0: "Not Survived", 1: "Survived"}
    )

    cabin_levels = sorted(tsne_df["Cabin"].unique())
    n_cabins = len(cabin_levels)
    if n_cabins <= 10:
        cabin_palette = sns.color_palette("tab10", n_colors=n_cabins)
    else:
        cabin_palette = sns.color_palette("tab20", n_colors=n_cabins)
    cabin_color_map = dict(zip(cabin_levels, cabin_palette))

    fig, ax = plt.subplots(figsize=(10, 7))
    marker = {"Not Survived": "X", "Survived": "o"}

    sns.scatterplot(
        data=tsne_df,
        x="TSNE-1",
        y="TSNE-2",
        hue="Cabin",
        style="Survived_lbl",
        markers=marker,
        palette=cabin_color_map,
        edgecolor="white",
        linewidth=0.5,
        s=80,
        alpha=0.85,
        ax=ax,
    )

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color="gray")
    ax.set_facecolor("#f8f9fa")

    plt.title(
        "t-SNE Visualization: Cabin (color) × Survival (marker)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncol=1,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9,
    )
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_cm_heatmap(cm: np.ndarray, acc: float, fig_path=None) -> None:
    """绘制混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        cbar=True,
        cbar_kws={"label": "计数", "shrink": 0.8},
        linewidths=1,
        linecolor="white",
        ax=ax,
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
    )
    plt.title(f"Confusion Matrix (Accuracy: {acc:.4%})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", rotation=0, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10, rotation=90)
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")


def plot_beatmap(model, X, y, feature_names, fig_path=None):
    """
    绘制特征基因组图谱 (Feature Genome Landscape)
    包含：特征共现矩阵(Dot) + 线性相关性(Top) + 模型重要性(Right)
    """
    pass


def main() -> None:
    """本示例主要展示如何针对缺失的离散信息和连续信息进行填补与可视化分析，学习本例前建议先看 Iris 可视化示例"""
    titanic_df = pd.read_csv(
        "D:\\Projects\\Coursework\\mcm_examples\\examples\\pd_examples\\data\\titanic.csv",
        header=0,
        sep=",",
        encoding="utf-8",
    )
    file_path = FIG_DIR / "titanic_heatmap.png"

    # highlight 1: 本例发现离散值 Cabin 缺失值太多，需要进行特殊处理
    titanic_df["Cabin"] = titanic_df["Cabin"].fillna("U")  # U - Unknown
    titanic_df["Cabin"] = titanic_df["Cabin"].apply(
        lambda x: x[0]
    )  # 只取首字母作为舱位编号，如 C185 - C

    # highlight 2: Age 作为存在缺失的连续值，可以使用 KNN 或者 MICE 插补
    imputer = KNNImputer(n_neighbors=4)
    imputed_cols = ["Age", "Fare", "Parch", "SibSp", "Pclass"]
    imputed = imputer.fit_transform(titanic_df[imputed_cols])
    titanic_df[imputed_cols] = imputed
    titanic_df["Age"] = pd.Series(
        np.rint(titanic_df["Age"].to_numpy()).astype(int), index=titanic_df.index
    )

    # 发现 Embarked 有极少量缺失，考虑直接丢弃
    titanic_df = titanic_df.dropna(subset=["Embarked"])

    all_features = titanic_df.columns.tolist().copy()
    features = titanic_df.drop(["PassengerId", "Name", "Ticket", "Survived"], axis=1)
    features = pd.get_dummies(
        features, drop_first=False
    )  # pd.get_dummies 用于离散值编码，避免距离计算错误
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # 解除注释以使用如下函数
    # compute_and_plot_tsne(X, titanic_df, fig_path=FIG_DIR / "titanic_tsne.png", perplexity=50)

    y = titanic_df["Survived"].to_numpy()
    le = LabelEncoder()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    param_grid = {
        "learning_rate": [0.089, 0.09, 0.091],
        "max_iter": [114, 115, 120],
        "max_depth": [3, 4, 5],
        "min_samples_leaf": [7, 8, 9],
        "l2_regularization": [1.1, 1.15, 1.2],
    }
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    gbt = HistGradientBoostingClassifier(random_state=42)
    gs = GridSearchCV(
        estimator=gbt,
        param_grid=param_grid,
        n_jobs=-1,
        scoring="accuracy",
        cv=inner_cv,
        return_train_score=True,
    )

    cv_results = cross_validate(
        gs,
        X_train,
        y_train_enc,
        cv=outer_cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1,
    )
    print(f"Test Scores: {cv_results["test_score"]}")
    print(
        f"Mean Test Score: {np.mean(cv_results['test_score']):.4%} ± {np.std(cv_results['test_score']):.4%}"
    )

    # highlight 经典套路：用 Grid Search 得到最好参数后，开始验证模型的泛化能力
    gs.fit(X_train, y_train_enc)
    print(f"Best Parameters: {gs.best_params_}")

    # 使用最佳参数创建新模型并训练（确保能获取 feature_importances_）
    best_params = gs.best_params_
    final_model = HistGradientBoostingClassifier(random_state=42, **best_params)
    final_model.fit(X_train, y_train_enc)

    # 在测试集上评估
    y_pred = final_model.predict(X_test)
    cm = confusion_matrix(y_test_enc, y_pred)
    acc = float(accuracy_score(y_test_enc, y_pred))
    print(f"Test Set Accuracy: {acc:.4%}")

    # 绘制特征相关性与重要性气泡图
    file_path = FIG_DIR / "titanic_beatmap.png"
    plot_beatmap(
        model=final_model,
        X=X_train,
        y=y_train_enc,
        feature_names=features.columns,
        fig_path=file_path,
    )


if __name__ == "__main__":
    main()
