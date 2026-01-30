import pandas as pd
import numpy as np

import re
from typing import Tuple

from configs.config import DATA_DIR, OUTPUT_DIR


def set_feature_finals(df: pd.DataFrame) -> pd.DataFrame:
    """
    标记决赛选手并计算每个赛季持续的周数

    处理逻辑：
    1. 识别 1st/2nd/3rd Place 的选手（决赛获奖选手）
    2. 计算每个赛季持续了多少周
    3. 标记进入决赛的选手
    """

    df["is_final_awarded"] = 0  # 是否获得前三名
    df["is_final_reached"] = 0  # 是否进入决赛
    df["season_total_weeks"] = 0  # 该赛季总共持续的周数
    df["weeks_participated"] = 0  # 该选手参与的周数

    week_cols = [
        col for col in df.columns if col.startswith("week") and col.endswith("_score")
    ]

    max_week = 0
    for col in week_cols:
        match = re.search(r"week(\d+)", col)
        if match:
            week_num = int(match.group(1))
            if week_num > max_week:
                max_week = week_num

    # 按赛季计算总周数
    season_weeks = {}
    for season in df["season"].unique():
        season_data = df[df["season"] == season]
        # 找到该赛季中任何选手最后一周有分数的周次
        max_week_in_season = 0
        for idx, row in season_data.iterrows():
            for week_num in range(1, max_week + 1):
                # 检查该周是否有任何评委给分
                week_score_cols = [
                    col for col in week_cols if col.startswith(f"week{week_num}_")
                ]
                if any(row[col] > 0 for col in week_score_cols if col in df.columns):
                    max_week_in_season = max(max_week_in_season, week_num)
        season_weeks[season] = max_week_in_season

    place_pattern = re.compile(r"^(1st|2nd|3rd)\s+Place$", re.IGNORECASE)

    for idx, row in df.iterrows():
        result = row["results"]
        season = row["season"]

        # 设置该赛季的总周数
        df.at[idx, "season_total_weeks"] = season_weeks.get(season, 0)

        # 计算该选手参与的周数
        participated_weeks = 0
        for week_num in range(1, max_week + 1):
            week_score_cols = [
                col for col in week_cols if col.startswith(f"week{week_num}_")
            ]
            if any(row[col] > 0 for col in week_score_cols if col in df.columns):
                participated_weeks = week_num
        df.at[idx, "weeks_participated"] = participated_weeks

        # 判断是否为前三名
        if pd.notna(result) and place_pattern.match(str(result)):
            df.at[idx, "is_final_awarded"] = 1
            df.at[idx, "is_final_reached"] = 1

            # 检查是否所有决赛选手都坚持到了最后一周
            if participated_weeks >= season_weeks.get(season, 0):
                pass
            else:
                pass

        # 对于未获奖但参与周数等于赛季总周数的选手，也标记为进入决赛
        elif (
            participated_weeks >= season_weeks.get(season, 0) and participated_weeks > 0
        ):
            df.at[idx, "is_final_reached"] = 1

        # 跳过退出的选手
        if pd.notna(result) and result == "withdrew":
            continue

    return df


def set_feature_low_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别每周评委分数较低但仍晋级的选手

    处理逻辑：
    1. 遍历每个赛季的每一周
    2. 计算该周所有参赛选手的平均分
    3. 使用箱线图方法计算Q1（下四分位数）
    4. 如果选手分数低于Q1但下一周仍有分数（说明晋级了），记录该周次
    5. 新增字段：
       - low_score_advanced_weeks: 所有低分晋级的周次列表（字符串形式，如"1,3,5"）
       - low_score_advanced_count: 低分晋级的总次数
       - low_score_advanced_week: 保留最后一次低分晋级的周次（向后兼容）
    """

    df["low_score_advanced_week"] = 0
    df["low_score_advanced_week"] = df["low_score_advanced_week"].astype("Int64")
    df["low_score_advanced_weeks"] = ""
    df["low_score_advanced_weeks"] = df["low_score_advanced_weeks"].astype("str")
    df["low_score_advanced_count"] = 0
    df["low_score_advanced_count"] = df["low_score_advanced_week"].astype("Int64")

    week_cols = [
        col for col in df.columns if col.startswith("week") and col.endswith("_score")
    ]

    max_week = 0
    for col in week_cols:
        match = re.search(r"week(\d+)", col)
        if match:
            week_num = int(match.group(1))
            if week_num > max_week:
                max_week = week_num

    # 根据每个赛季的每个周处理相关的代码
    for season in df["season"].unique():
        season_data = df[df["season"] == season]
        total_weeks = season_data["season_total_weeks"].iloc[0]

        if total_weeks <= 1:
            continue

        for week_num in range(1, total_weeks):
            week_score_cols = [
                col for col in week_cols if col.startswith(f"week{week_num}_")
            ]

            week_scores = []
            week_indices = []

            for idx, row in season_data.iterrows():
                valid_scores = [row[col] for col in week_score_cols if row[col] > 0]

                if len(valid_scores) > 0:
                    avg_score = np.mean(valid_scores)
                    week_scores.append(avg_score)
                    week_indices.append((idx, avg_score))

            if len(week_scores) < 4:
                continue

            q1 = np.percentile(week_scores, 25)

            next_week_score_cols = [
                col for col in week_cols if col.startswith(f"week{week_num + 1}_")
            ]

            for idx, score in week_indices:
                if score < q1:
                    row = df.loc[idx]
                    next_week_valid = any(row[col] > 0 for col in next_week_score_cols)

                    if next_week_valid:
                        df.at[idx, "low_score_advanced_week"] = week_num

                        # 追加到所有低分晋级周次列表
                        current_weeks = df.at[idx, "low_score_advanced_weeks"]
                        current_weeks = str(current_weeks)
                        if current_weeks == "":
                            df.at[idx, "low_score_advanced_weeks"] = str(week_num)
                        else:
                            df.at[idx, "low_score_advanced_weeks"] = (
                                current_weeks + "," + str(week_num)
                            )

                        df.at[idx, "low_score_advanced_count"] += 1

    return df


def handle_data(file_name: str) -> pd.DataFrame:
    """进行 2026C 题的数据初步处理"""
    try:
        file_path = DATA_DIR / file_name
        df = pd.read_csv(file_path, sep=",", header=0, encoding="utf-8")
        obj_cols = [
            "celebrity_name",
            "ballroom_partner",
            "celebrity_industry",
            "celebrity_homestate",
            "celebrity_homecountry/region",
            "results",
        ]
        float_cols = df.select_dtypes(include=["float64"]).columns.tolist()
        df[float_cols] = df[float_cols].fillna(0.0)
        for col in obj_cols:
            df[col] = df[col].astype("string")

        # highlight: 对选手身份的特征分析
        df["is_from_usa"] = 0
        for col in df.index:
            country = df.at[col, "celebrity_homecountry/region"]
            if pd.notna(country) and country == "United States":
                df.at[col, "is_from_usa"] = 1

        return df

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise


def data_sparse(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    season_3_to_27 = df[df["season"].between(3, 27)]
    remaining = df[~df["season"].between(3, 27)]
    return season_3_to_27, remaining


def set_feature_score_sum(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每一周所有评委给选手的投票总和

    为每一周添加新列：week{N}_score_sum
    """
    week_cols = [
        col for col in df.columns if col.startswith("week") and col.endswith("_score")
    ]

    week_nums = set()
    for col in week_cols:
        match = re.search(r"week(\d+)", col)
        if match:
            week_nums.add(int(match.group(1)))

    for week_num in sorted(week_nums):
        week_score_cols = [
            col for col in week_cols if col.startswith(f"week{week_num}_")
        ]
        df[f"week{week_num}_score_sum"] = df[week_score_cols].sum(axis=1)

    return df


def set_feature_rank(df: pd.DataFrame, *, method="rank") -> pd.DataFrame:
    """
    根据指定方法计算排名特征，采用工厂模式实现排名的设计。

    method="rank": 基于每周总票数的排名，添加 week{N}_judge_score_rank
    method="percentage": 基于每周票数占比的排名，添加 week{N}_percentage 和 week{N}_percentage_rank
    """
    week_cols = [
        col
        for col in df.columns
        if col.startswith("week") and col.endswith("_score_sum")
    ]

    if method == "rank":
        # 为每一周添加排名，基于 week{N}_score_sum 降序排名
        for col in week_cols:
            week_num = col.replace("week", "").replace("_score_sum", "")
            rank_col = f"week{week_num}_judge_score_rank"
            df[rank_col] = 0
            mask = df[col] > 0
            if mask.any():
                df.loc[mask, rank_col] = (
                    df[mask]
                    .groupby("season")[col]
                    .rank(method="dense", ascending=False)
                    .astype(int)
                )
    elif method == "percentage":
        # 计算每周总票数
        for season in df["season"].unique():
            season_mask = df["season"] == season
            season_data = df[season_mask]
            for col in week_cols:
                week_num = col.replace("week", "").replace("_score_sum", "")
                total_votes = season_data[col].sum()
                percentage_col = f"week{week_num}_percentage"
                rank_col = f"week{week_num}_percentage_rank"
                if total_votes > 0:
                    df.loc[season_mask, percentage_col] = (
                        df.loc[season_mask, col] / total_votes
                    )
                    mask = (df["season"] == season) & (df[col] > 0)
                    df.loc[mask, rank_col] = (
                        df.loc[mask, percentage_col]
                        .rank(method="dense", ascending=False)
                        .astype(int)
                    )
                    df.loc[~mask & season_mask, rank_col] = 0
                else:
                    df.loc[season_mask, percentage_col] = 0.0
                    df.loc[season_mask, rank_col] = 0

    return df


def main():
    df = handle_data("2026_MCM_Problem_C_Data.csv")
    df = set_feature_finals(df)
    df = set_feature_low_score(df)
    df = set_feature_score_sum(df)

    # 应用 rank 方法
    df_rank = df.copy()
    df_rank = set_feature_rank(df_rank, method="rank")

    # 应用 percentage 方法
    df_percentage = df.copy()
    df_percentage = set_feature_rank(df_percentage, method="percentage")

    rank_df, percentage_df = data_sparse(df)

    output_path = OUTPUT_DIR / "processed_data.csv"
    output_path_rank = OUTPUT_DIR / "processed_data_rank.csv"
    output_path_percentage = OUTPUT_DIR / "processed_data_percentage.csv"

    df.to_csv(output_path, index=False)
    df_rank.to_csv(output_path_rank, index=False)
    df_percentage.to_csv(output_path_percentage, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Rank version saved to: {output_path_rank}")
    print(f"Percentage version saved to: {output_path_percentage}")


if __name__ == "__main__":
    main()
