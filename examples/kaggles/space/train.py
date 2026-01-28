import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint

import joblib
import json

from configs.config import DATA_DIR, OUTPUT_DIR
from feature import add_features


def xgb_train(X_train, y_train):
    """基于 XGBoost 的分类预测模型的预处理和模型训练，最终返回最佳模型的 Pipeline"""
    numeric_cols = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalBill",
    ]
    onehot_cols = ["Destination", "HomePlanet"]
    ordinal_cols = ["Deck", "Side"]

    xgbc = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    numeric_preprocessor = Pipeline(
        [
            ("knn", KNNImputer()),
            ("scaler", StandardScaler()),
        ]
    )
    onehot_preprocessor = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    label_preprocessor = Pipeline(
        [
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            )
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocessor, numeric_cols),
            ("cat", onehot_preprocessor, onehot_cols),
            (
                "ordinal",
                label_preprocessor,
                ordinal_cols,
            ),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", xgbc),
        ]
    )
    param_dist = {
        "preprocessor__num__knn__n_neighbors": randint(3, 10),
        "preprocessor__num__knn__weights": ["uniform", "distance"],
        "classifier__n_estimators": randint(100, 500),
        "classifier__max_depth": randint(3, 10),
        "classifier__learning_rate": uniform(0.01, 0.3),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__gamma": uniform(0, 5),
    }
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="accuracy",
        cv=inner_cv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    rs.fit(X_train, y_train)
    return rs.best_estimator_


def main():
    file = OUTPUT_DIR / "processed_train.csv"
    df = pd.read_csv(file, sep=",", header=0, encoding="utf-8")
    df = add_features(df)  # 先添加特征
    X = df.drop(columns=["Transported", "Group"])
    y = df["Transported"].astype(bool)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    best_model = xgb_train(X_train, y_train)

    # 用最佳模型做 CV
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score),
    }
    cv_result = cross_validate(
        best_model,
        X_train,
        y_train,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    print("Cross-validation results:")
    print(
        f"Train Accuracy: {cv_result['train_accuracy'].mean():.4f} ± {cv_result['train_accuracy'].std():.4f}"
    )
    print(
        f"Test Accuracy: {cv_result['test_accuracy'].mean():.4f} ± {cv_result['test_accuracy'].std():.4f}"
    )
    print(
        f"Train F1: {cv_result['train_f1'].mean():.4f} ± {cv_result['train_f1'].std():.4f}"
    )
    print(
        f"Test F1: {cv_result['test_f1'].mean():.4f} ± {cv_result['test_f1'].std():.4f}"
    )

    # 保存最佳参数到 JSON
    from sklearn.pipeline import Pipeline

    if isinstance(best_model, Pipeline):
        best_params = best_model.named_steps["classifier"].get_params()
    else:
        best_params = best_model.get_params()  # fallback if not Pipeline
    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to {OUTPUT_DIR / 'best_params.json'}")

    if hasattr(best_model, "score"):
        test_score = best_model.score(X_test, y_test)
        print(f"Holdout test accuracy: {test_score:.4f}")
    else:
        print("Model does not have score method, skipping holdout evaluation")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, OUTPUT_DIR / "best_xgb_pipeline.joblib")
    print(f"Model saved to {OUTPUT_DIR / 'best_xgb_pipeline.joblib'}")


if __name__ == "__main__":
    main()
