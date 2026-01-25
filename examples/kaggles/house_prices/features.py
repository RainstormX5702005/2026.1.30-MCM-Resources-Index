import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV, cross_validate, train_test_split
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
        df = pd.read_csv(data_path, sep=',', header=0, encoding='utf-8')
        
        # highlight: 对数值型特征使用基于 'distance' 的 KNN 算法填充缺失值
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        imputed = imputer.fit_transform(df[numeric_cols])
        imputed = scaler.inverse_transform(imputed)
        df[numeric_cols] = imputed  # warning: 回归模型应该逆标准化得到原数据

        # 对所有非数值列直接用 LabelEncoder（简化版，适合 XGBoost）
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
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
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
    X = df.drop(columns=['Id', 'SalePrice'])
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),

        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 3)
    }

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator=xgbr,
        param_distributions=param_dist,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=inner_cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
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

    # 评估：若已有保存模型则加载，否则用简单 XGB 在全量数据上训练并保存
    X = df.drop(columns=['Id', 'SalePrice'])
    y = df['SalePrice']

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("No saved model found — training a default XGBRegressor on full data...")
        model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")

    # 打印模型参数
    print('\nModel parameters:')
    for k, v in model.get_params().items():
        print(f"  {k}: {v}")

    # 使用 K-Fold 交叉验证得到每折 MSE、均值和标准差
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    neg_mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
    mse_scores = -neg_mse_scores
    print('\nPer-fold MSE:')
    for i, s in enumerate(mse_scores, 1):
        print(f"  Fold {i}: {s:.6f}")
    print(f"CV MSE mean: {mse_scores.mean():.6f}")
    print(f"CV MSE std : {mse_scores.std():.6f}")

    # 计算并打印 CV 上的 R^2
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf, n_jobs=-1)
    print('\nPer-fold R2:')
    for i, s in enumerate(r2_scores, 1):
        print(f"  Fold {i}: {s:.6f}")
    print(f"CV R2 mean: {r2_scores.mean():.6f}")
    print(f"CV R2 std : {r2_scores.std():.6f}")

    # 在全量数据上做个训练集 MSE 查看
    preds = model.predict(X)
    mse_train = mean_squared_error(y, preds)
    print(f"\nTrain MSE: {mse_train:.6f}")
    r2_train = r2_score(y, preds)
    print(f"Train R2: {r2_train:.6f}")

if __name__ == "__main__":
    main()