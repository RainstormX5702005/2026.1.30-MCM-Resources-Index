import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from configs.config import DATA_DIR, OUTPUT_DIR


def preprocessing(file_name: str) -> pd.DataFrame:
    """进行部分数据的标准化和编码处理"""
    file_path = OUTPUT_DIR / file_name
    df = pd.read_csv(file_path, sep=",", header=0, encoding="utf-8")

    processed_df = df.copy()
    oe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    le = LabelEncoder()
    scaler = StandardScaler()

    processed_df["advanced_rounds"] = processed_df["low_score_advanced_week"] - 1

    standard_cols = [
        "celebrity_age_during_season",
        "advanced_rounds",
    ]

    week_sum_cols = [col for col in processed_df.columns if col.endswith("_score_sum")]
    standard_cols.extend(week_sum_cols)

    processed_df[standard_cols] = scaler.fit_transform(processed_df[standard_cols])

    encoded_industry = oe.fit_transform(processed_df[["celebrity_industry"]])
    encoded_df = pd.DataFrame(
        encoded_industry, columns=oe.get_feature_names_out(["celebrity_industry"])
    )
    processed_df = pd.concat([processed_df, encoded_df], axis=1)
    processed_df.drop("celebrity_industry", axis=1, inplace=True)

    processed_df["celebrity_name"] = le.fit_transform(processed_df["celebrity_name"])
    return processed_df


def main():
    processed_percentage_df = preprocessing("processed_data_percentage.csv")
    processed_percentage_path = (
        OUTPUT_DIR / "preprocessed" / "preprocessed_data_percentage.csv"
    )
    processed_percentage_df.to_csv(processed_percentage_path, index=False)

    preprocessed_rank_df = preprocessing("processed_data_rank.csv")
    preprocessed_rank_path = OUTPUT_DIR / "preprocessed" / "preprocessed_data_rank.csv"
    preprocessed_rank_df.to_csv(preprocessed_rank_path, index=False)

    print(f"Preprocess finished successfully!")


if __name__ == "__main__":
    main()
