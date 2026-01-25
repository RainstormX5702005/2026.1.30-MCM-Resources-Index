import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    GridSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.impute import KNNImputer
import xgboost as xgb
from scipy.stats import uniform, randint

from configs.config import DATA_DIR
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

MODEL_PATH = DATA_DIR / "xgb_model.joblib"


def handle_data(file_name: str) -> tuple[pd.DataFrame, StandardScaler]:
    """进行房屋价格的数据处理训练"""
    try:
        data_path = DATA_DIR / file_name
        df = pd.read_csv(data_path, sep=",", header=0, encoding="utf-8")

        # highlight: 对数值型特征使用基于 'distance' 的 KNN 算法填充缺失值
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        imputed = imputer.fit_transform(df[numeric_cols])
        imputed = scaler.inverse_transform(imputed)
        df[numeric_cols] = imputed  # warning: 回归模型应该逆标准化得到原数据

        # 对所有非数值列直接用 LabelEncoder（简化版，适合 XGBoost）
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].fillna("NA")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        return df, scaler

    except FileNotFoundError:
        print(f"File {file_name} not found in {DATA_DIR}")
        return pd.DataFrame(), StandardScaler()


def xgb_model_train(df: pd.DataFrame):
    """使用 XGBoost 做模型的训练"""
    xgbr = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42)
    X = df.drop(columns=["Id", "SalePrice"])
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    param_dist = {
        "n_estimators": randint(100, 1000),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 3),
    }

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator=xgbr,
        param_distributions=param_dist,
        n_iter=100,
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )
    rs.fit(X_train, y_train)
    # 保存最佳模型方便后续直接加载（如果需要）
    best = rs.best_estimator_
    try:
        joblib.dump(best, MODEL_PATH)
        print(f"Saved best model to {MODEL_PATH}")
    except Exception:
        pass
    return rs.best_estimator_


def main():
    df, scaler = handle_data("train.csv")
    save_path = DATA_DIR / "preprocessed_train.csv"
    df.to_csv(save_path, index=False)

    # 直接在main中完成训练集和测试集的划分
    X = df.drop(columns=["Id", "SalePrice"])
    y = df["SalePrice"]

    print("=" * 60)
    print("划分训练集和测试集 (70% train, 30% test)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")

    # 检查是否有已保存的模型
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"\n已加载模型: {MODEL_PATH}")
    else:
        print("\n未找到已保存的模型，开始训练新模型...")
        model = xgb_model_train(df)
        print(f"已保存模型到: {MODEL_PATH}")

    # 打印模型参数
    print("\n" + "=" * 60)
    print("模型参数:")
    print("=" * 60)
    for k, v in model.get_params().items():
        print(f"  {k}: {v}")

    # 在训练集上评估
    print("\n" + "=" * 60)
    print("训练集表现:")
    print("=" * 60)
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Train MSE:  {train_mse:.6f}")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Train R²:   {train_r2:.6f}")

    # 在测试集上评估（关键！）
    print("\n" + "=" * 60)
    print("测试集表现:")
    print("=" * 60)
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Test MSE:  {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test R²:   {test_r2:.6f}")

    # 过拟合检测
    print("\n" + "=" * 60)
    print("过拟合分析:")
    print("=" * 60)
    print(f"R² 差值 (Train - Test): {train_r2 - test_r2:.6f}")
    print(f"RMSE 比率 (Test / Train): {test_rmse / train_rmse:.4f}")
    if test_r2 < train_r2 - 0.1:
        print("⚠️  警告: 模型可能存在过拟合")
    else:
        print("✓ 模型泛化能力良好")

    # 交叉验证作为额外参考
    print("\n" + "=" * 60)
    print("5折交叉验证 (在全数据集上):")
    print("=" * 60)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse_scores = -cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1
    )
    cv_r2_scores = cross_val_score(model, X, y, scoring="r2", cv=kf, n_jobs=-1)
    print(f"CV MSE:  {cv_mse_scores.mean():.6f} ± {cv_mse_scores.std():.6f}")
    print(f"CV RMSE: {np.sqrt(cv_mse_scores.mean()):.6f}")
    print(f"CV R²:   {cv_r2_scores.mean():.6f} ± {cv_r2_scores.std():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
