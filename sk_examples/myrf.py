from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    StratifiedGroupKFold,
)

from sklearn.preprocessing import LabelEncoder

import pandas as pd

import numpy as np


def main():
    """使用随机森林分类器对 Iris 数据集进行分类，并使用 cross-validation 与 grid search 评价模型准确性与稳定性。"""

    try:

        # 使用 `pandas` 读取 Iris 数据集

        iris_df = pd.read_csv(
            "D:\\Projects\\Coursework\\mcm_examples\\pd_examples\\data\\Iris.csv",
            header=0,
            sep=",",
            encoding="utf-8",
        )

        species = iris_df["Species"].unique()

        mapping = {label: idx for idx, label in enumerate(species)}

        iris_df["Species"] = iris_df["Species"].map(mapping)

        # 将预处理完成的数据转为机器学习的输入格式

        X = iris_df.drop(["Id", "Species"], errors="ignore").values

        y = iris_df["Species"].values

        le = LabelEncoder()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=np.asarray(y)
        )

        y_train_enc = le.fit_transform(y_train)

        y_test_enc = le.transform(y_test)

        param_grid = {
            "n_estimators": [5, 10, 20, 30, 40, 50],
            "max_depth": [None, 5, 10, 15, 20],
            "max_features": ["sqrt", "log2"],
        }

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # highlight: 使用 Random Forest 分类器来判断花的种类，并采用 gs 调参

        gs = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            n_jobs=-1,
            scoring="accuracy",
            cv=inner_cv,
        )

        gs.fit(X_train, y_train_enc)

        best_params = gs.best_params_

    except FileNotFoundError:

        print("File not found: Iris.csv")

        return


if __name__ == "__main__":

    main()
