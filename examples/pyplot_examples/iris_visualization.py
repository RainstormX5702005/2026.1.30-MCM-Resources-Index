from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    accuracy_score,
)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from configs.encode import set_encoding_method, FIG_DIR


set_encoding_method()


def main():
    """使用随机森林分类器对 Iris 数据集进行分类，并使用 cross-validation 与 grid search 评价模型准确性与稳定性。"""
    try:
        # highlight: 以下代码来自于 ./sk_examples/rf.py

        # 读取 Iris 数据集，用 `pandas` 即可。
        iris_df = pd.read_csv(
            "D:\\Projects\\Coursework\\mcm_examples\\examples\\pd_examples\\data\\Iris.csv",
            header=0,
            sep=",",
            encoding="utf-8",
        )

        features = iris_df.drop(["Id", "Species"], axis=1, errors="ignore")
        features = features.apply(pd.to_numeric, errors="coerce").fillna(
            features.mean()
        )
        X = features.to_numpy(dtype=float)

        # fixed: 使用 LabelEncoder 编码标签
        le = LabelEncoder()  # warning: 产生错误的原因是因为当时没正确删掉列，现已解决。
        y_raw = iris_df["Species"].astype(str).to_numpy()
        y_enc = le.fit_transform(y_raw)

        # 为了评估模型是否准确，训练集与测试集十分重要，需要调用 `train_test_split`
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
        )

        # GRID SEARCH Hyperparams 空间设置，需要根据样本数目设定可能的空间
        param_grid = {
            "n_estimators": [20, 50, 100, 200],
            "max_depth": [2, 3, 5, 10, None],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        }
        # 外层 cv 用于评价整体模型
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # 内层 cv 用于 Hyperparams 搜索
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        gs = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            n_jobs=-1,
            scoring="accuracy",
            cv=inner_cv,
            verbose=1,
        )

        gs.fit(X_train, y_train)
        # warning: 此处不用 cross_val_predict 的理由是 —— 我们需要用测试集评价模型，而交叉验证法都是在训练集中评估

        # highlight: 从此处开始为新代码，用于可视化模型的评价结果
        y_pred = gs.predict(X_test)  # highlight: 此处开始使用测试集来评价模型是否准确
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average="macro")
        pcs = precision_score(y_test, y_pred, average="macro")

        cm = confusion_matrix(y_test, y_pred)

        # 此处开始绘图 — 使用 seaborn 绘制特征相关性热力图（替代原混淆矩阵可视化）
        corr = features.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            linecolor="black",
            cbar_kws={"label": "相关系数"},
            ax=ax,
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Iris 特征相关性热力图")

        # 在图上保留模型评价摘要（可选）
        txt = f"准确率 ACU: {acc:.4%}\n召回率 REC: {rec:.4%}\n精确率 PCS: {pcs:.4%}"
        ax.annotate(
            text=txt,
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="#5916F5", alpha=0.5),
            fontsize=12,
        )

        plt.tight_layout()
        plt.show()

        # warning: 新代码段结束

    except FileNotFoundError:
        print("File or Directory not found: {e.filename}")
        return


if __name__ == "__main__":
    main()
