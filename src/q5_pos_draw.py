import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from configs.config import DATA_DIR, OUTPUT_DIR


def rebuild_df(pct_df: pd.DataFrame, rank_df: pd.DataFrame) -> pd.DataFrame:
    """实现两个 df 的重构与合并，方便后续操作"""
    pct_df_processed = pct_df.copy()
    pct_df_processed.columns = [
        "Season",
        "Week",
        "Celebrity_Name",
        "Predicted_Elim_Probability",  # 原来是Prob_Advancement
        "Actual_Eliminated",  # 原来是Actually_Advanced
    ]

    # Actually_Advanced: 1表示晋级，0表示淘汰 -> 需要反转为 Actual_Eliminated
    pct_df_processed["Actual_Eliminated"] = (
        pct_df_processed["Actual_Eliminated"] == 0
    ).astype(bool)

    # Prob_Advancement 是晋级概率 -> 转换为淘汰概率
    pct_df_processed["Predicted_Elim_Probability"] = (
        1.0 - pct_df_processed["Predicted_Elim_Probability"]
    )

    pct_df_processed["Prediction_Type"] = "pct"

    # 处理 rank_df - 保留原始浮点数精度
    rank_df_processed = rank_df.copy()
    rank_df_processed = rank_df_processed.drop(columns=["Contestant_ID"])
    rank_df_processed.columns = [
        "Season",
        "Week",
        "Celebrity_Name",
        "Actual_Eliminated",
        "Predicted_Elim_Probability",
    ]

    # 重新排列列的顺序，使其与pct_df一致
    rank_df_processed = rank_df_processed[
        [
            "Season",
            "Week",
            "Celebrity_Name",
            "Predicted_Elim_Probability",
            "Actual_Eliminated",
        ]
    ]

    # Actual_Eliminated 已经是正确的：0表示未淘汰，1表示淘汰
    rank_df_processed["Actual_Eliminated"] = rank_df_processed[
        "Actual_Eliminated"
    ].astype(bool)

    rank_df_processed["Prediction_Type"] = "rank"

    # 合并两个数据集
    res_df = pd.concat([pct_df_processed, rank_df_processed], ignore_index=True)
    res_df = res_df.sort_values(by=["Season", "Week", "Celebrity_Name"]).reset_index(
        drop=True
    )

    return res_df


def main():
    input_dir = OUTPUT_DIR / "question1_res" / "draw_await"
    output_dir = OUTPUT_DIR / "question5_res"

    org_df = pd.read_csv(
        OUTPUT_DIR / "q4_featured_data",
        sep=",",
        header=0,
        encoding="utf-8",
    )

    pct_df = pd.read_csv(
        input_dir / "percent" / "pct_advancement_preds.csv",
        sep=",",
        header=0,
        encoding="utf-8",
        dtype={
            "Season": "int64",
            "Week": "int64",
            "Name": "str",
            "Prob_Advancement": "float64",
            "Actually_Advanced": "int64",
        },
    )

    rank_df = pd.read_csv(
        input_dir / "rank" / "rank_elimination_probabilities.csv",
        sep=",",
        header=0,
        encoding="utf-8",
        dtype={
            "Season": "int64",
            "Week": "int64",
            "Contestant_ID": "int64",
            "Celebrity_Name": "str",
            "Actual_Eliminated": "int64",
            "Predicted_Elim_Probability": "float64",
        },
    )

    combined_df = rebuild_df(pct_df, rank_df)
    combined_df.to_csv(output_dir / "q5_combined.csv", index=False)


if __name__ == "__main__":
    main()
