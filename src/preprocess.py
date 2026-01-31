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
    processed_df[standard_cols] = scaler.fit_transform(processed_df[standard_cols])

    week_sum_cols = [col for col in processed_df.columns if col.endswith("_score_sum")]
    for season in processed_df["season"].unique():
        season_mask = processed_df["season"] == season
        for col in week_sum_cols:
            valid_mask = season_mask & (processed_df[col] > 0)
            if valid_mask.sum() > 1:  # 至少需要2个有效值才能标准化
                scaler_week = StandardScaler()
                processed_df.loc[valid_mask, col] = scaler_week.fit_transform(
                    pd.to_numeric(processed_df.loc[valid_mask, col])
                    .to_numpy()
                    .reshape(-1, 1)
                ).flatten()

    encoded_industry = oe.fit_transform(processed_df[["celebrity_industry"]])
    encoded_df = pd.DataFrame(
        encoded_industry, columns=oe.get_feature_names_out(["celebrity_industry"])
    )
    processed_df = pd.concat([processed_df, encoded_df], axis=1)
    processed_df.drop("celebrity_industry", axis=1, inplace=True)

    week_score_cols = [
        col
        for col in processed_df.columns
        if col.startswith("week")
        and col.endswith("_score")
        and not col.endswith("_score_sum")
    ]
    processed_df.drop(columns=week_score_cols, inplace=True)

    columns_to_drop = [
        "ballroom_partner",
        "celebrity_homecountry/region",
        "results",
        "celebrity_homestate",
    ]
    processed_df.drop(columns=columns_to_drop, inplace=True)

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
