from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def main():
    """使用随机森林分类器对 Iris 数据集进行分类，并使用 cross-validation 与 grid search 评价模型准确性与稳定性。"""
    try:
        # 读取 Iris 数据集，用 `pandas` 即可。
        iris_df = pd.read_csv(
            "D:\\Projects\\Coursework\\mcm_examples\\pd_examples\\data\\Iris.csv",
            header=0,
            sep=",",
            encoding="utf-8",
        )

        features = iris_df.drop(["Id", "Species"], axis=1, errors="ignore")
        features = features.apply(pd.to_numeric, errors="coerce").fillna(
            features.mean()
        )
        X = features.to_numpy(dtype=float)

        # question: 不知为何用 LabelEncoder 会报错，故使用手动编码
        y_raw = iris_df["Species"].astype(str).to_numpy()
        uniques = pd.unique(y_raw)
        mapping = {label: idx for idx, label in enumerate(uniques)}
        y_enc = pd.Series(y_raw).map(mapping).to_numpy(dtype=int)

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

        print(f"\nBest Hyperparams: {gs.best_params_}")
        print(f"Best Cross Validate: {gs.best_score_:.4f}\n")

        test_score = gs.score(X_test, y_test)
        print(f"ACU: {test_score:.4f}\n")

        nested_cv_scores = cross_validate(
            gs, X_train, y_train, cv=outer_cv, scoring="accuracy"
        )

        print(f"MEAN: {nested_cv_scores['test_score'].mean():.4f}")
        print(f"STD2: {nested_cv_scores['test_score'].var():.4f}")
        print(f"STD: {nested_cv_scores['test_score'].std():.4f}")

    except FileNotFoundError:
        print("File not found: Iris.csv")
        return


if __name__ == "__main__":
    main()
