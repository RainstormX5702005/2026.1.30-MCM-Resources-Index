import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    train_test_split,
    cross_validate,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from scipy.stats import uniform, randint

from configs.config import DATA_DIR
import joblib
from sklearn.metrics import mean_squared_error, r2_score

MODEL_PATH = DATA_DIR / "xgb_model.joblib"


def handle_data(file_name: str) -> tuple[pd.DataFrame, StandardScaler]:
    """读取并做最小清洗；不要在这里 fit 任何会泄露信息的预处理器。

    返回 (df, placeholder_scaler) — scaler 仅为兼容旧接口。
    """
    try:
        data_path = DATA_DIR / file_name
        df = pd.read_csv(data_path, sep=",", header=0, encoding="utf-8")

        # 只做最小的类目空值填充；数值缺失保留为 NaN，由训练时的 imputer 处理
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].fillna("NA")

        return df, StandardScaler()

    except FileNotFoundError:
        print(f"File {file_name} not found in {DATA_DIR}")
        return pd.DataFrame(), StandardScaler()


def xgb_model_train(df: pd.DataFrame):
    """使用 XGBoostRegressor 模型来训练房价预测模型"""
    xgbr = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42)
    X = df.drop(columns=["Id", "SalePrice"])
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 划分数值与类别特征，用 ColumnTransformer 在训练时 fit imputer/scaler
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipeline = Pipeline(
        [
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
            ("scaler", StandardScaler()),
            ("pca", PCA(svd_solver="full", random_state=42)),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
            (
                "enc",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, numeric_cols), ("cat", cat_pipeline, categorical_cols)],
        remainder="drop",
    )

    pipeline = Pipeline([("pre", preprocessor), ("xgbr", xgbr)])

    param_dist = {
        "pre__num__pca__n_components": [0.95, 0.96, 0.99, "mle", None],
        "xgbr__n_estimators": randint(100, 1000),
        "xgbr__max_depth": randint(3, 10),
        "xgbr__learning_rate": uniform(0.01, 0.2),
        "xgbr__subsample": uniform(0.6, 0.4),
        "xgbr__colsample_bytree": uniform(0.6, 0.4),
        "xgbr__reg_alpha": uniform(0, 1),
        "xgbr__reg_lambda": uniform(0, 3),
    }

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=100,
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    rs.fit(X_train, y_train)

    # 在训练集上对最优模型做交叉验证，报告 MSE 与 R2 的均值与标准差
    cv_results = cross_validate(
        rs.best_estimator_,
        X_train,
        y_train,
        cv=inner_cv,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        n_jobs=-1,
        return_train_score=False,
    )
    mse_scores = -cv_results["test_mse"]
    r2_scores = cv_results["test_r2"]
    mse_mean, mse_std = mse_scores.mean(), mse_scores.std(ddof=1)
    r2_mean, r2_std = r2_scores.mean(), r2_scores.std(ddof=1)
    print(
        f"CV MSE: mean={mse_mean:.4f}, std={mse_std:.4f}; CV R2: mean={r2_mean:.4f}, std={r2_std:.4f}"
    )

    y_pred = rs.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")
    # 保存最佳模型方便后续直接加载（如果需要）
    best = rs.best_estimator_
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, MODEL_PATH)
        print(f"Saved best model to {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to save model: {e}")
    return mse, r2, mse_mean, mse_std, r2_mean, r2_std, rs.best_estimator_


def main():
    df, _ = handle_data("house_prices.csv")
    if df.empty:
        print("No data to train on.")
        return

    (
        mse,
        r2,
        mse_mean,
        mse_std,
        r2_mean,
        r2_std,
        best_model,
    ) = xgb_model_train(df)

    print("Training complete.")
    print(f"Test MSE: {mse:.4f}, Test R2: {r2:.4f}")
    print(f"CV MSE: mean={mse_mean:.4f}, std={mse_std:.4f}")
    print(f"CV R2: mean={r2_mean:.4f}, std={r2_std:.4f}")


if __name__ == "__main__":
    main()
