import pandas as pd
import joblib
from pathlib import Path

from configs.config import DATA_DIR, OUTPUT_DIR
from feature import handle_data, add_features


def predict_test():
    # 加载训练好的模型
    model_path = OUTPUT_DIR / "best_xgb_pipeline.joblib"
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # 加载原始测试数据以获取 PassengerId
    test_raw = pd.read_csv(DATA_DIR / "test.csv", sep=",", header=0, encoding="utf-8")
    passenger_ids = test_raw["PassengerId"]

    test_df = handle_data("test.csv")
    test_df = add_features(test_df)

    predictions = model.predict(test_df)
    predictions_proba = model.predict_proba(test_df)[:, 1]  # 如果需要概率

    submission = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "Transported": predictions.astype(bool),
        }
    )

    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # 可选：打印一些统计
    print(f"Predicted Transported: {predictions.sum()} out of {len(predictions)}")


if __name__ == "__main__":
    predict_test()
