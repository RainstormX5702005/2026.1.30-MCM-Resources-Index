import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from configs.config import DATA_DIR

MODEL_NAME = "xgb_model.joblib"


def predict_file(
    model_path: Path, input_csv: Path, out_csv: Path, smoke: int | None = None
):

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading pipeline from {model_path}...")
    pipeline = joblib.load(model_path)

    df = pd.read_csv(input_csv)

    ids = df["Id"] if "Id" in df.columns else pd.RangeIndex(len(df))

    X = df.drop(columns=["Id", "SalePrice"], errors="ignore")

    if smoke is not None:
        X = X.head(smoke)
        ids = ids[: len(X)]

    print(f"Predicting on shape: {X.shape}...")

    preds = pipeline.predict(X)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"Id": ids, "SalePrice": preds})
    submission.to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv}")


def main():
    p = argparse.ArgumentParser(description="Predict using saved pipeline model")
    p.add_argument("--model", type=Path, default=DATA_DIR / MODEL_NAME)
    p.add_argument("--input", type=Path, default=DATA_DIR / "test.csv")
    p.add_argument("--output", type=Path, default=DATA_DIR / "submission.csv")
    p.add_argument(
        "--smoke", type=int, default=None, help="If set, only predict first N rows"
    )
    args = p.parse_args()

    predict_file(args.model, args.input, args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
