import pandas as pd
import numpy as np

import re
from typing import Tuple

from configs.config import DATA_DIR, OUTPUT_DIR


def set_final(df: pd.DataFrame) -> pd.DataFrame:
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

    # 获取所有week_judge_score列
    week_cols = [
        col for col in df.columns if col.startswith("week") and col.endswith("_score")
    ]

    # 提取周数信息（从列名中获取）
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
    3. 使用箱型图方法计算Q1（下四分位数）
    4. 如果选手分数低于Q1但下一周仍有分数（说明晋级了），记录该周次
    5. 新增字段：
       - low_score_advanced_weeks: 所有低分晋级的周次列表（字符串形式，如"1,3,5"）
       - low_score_advanced_count: 低分晋级的总次数
       - low_score_advanced_week: 保留最后一次低分晋级的周次（向后兼容）
    """

    df["low_score_advanced_week"] = 0
    df["low_score_advanced_weeks"] = ""
    df["low_score_advanced_weeks"] = df["low_score_advanced_weeks"].astype("str")
    df["low_score_advanced_count"] = 0

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

    # 按赛季处理
    for season in df["season"].unique():
        season_data = df[df["season"] == season]
        total_weeks = season_data["season_total_weeks"].iloc[0]

        if total_weeks <= 1:
            continue

        for week_num in range(1, total_weeks):
            # 获取该周的所有评委分数列
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
                        if current_weeks == "":
                            df.at[idx, "low_score_advanced_weeks"] = str(week_num)
                        else:
                            df.at[idx, "low_score_advanced_weeks"] = (
                                current_weeks + "," + str(week_num)
                            )

                        # 增加计数
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

        df["is_from_usa"] = 0
        for col in df.index:
            country = df.at[col, "celebrity_homecountry/region"]
            if pd.notna(country) and country == "United States":
                df.at[col, "is_from_usa"] = 1

        return df

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise


def main():
    df = handle_data("2026_MCM_Problem_C_Data.csv")
    df = set_final(df)
    df = set_feature_low_score(df)

    output_path = OUTPUT_DIR / "processed_data.csv"

    print("\n=== Statistics ===")
    print(f"Total contestants: {len(df)}")
    print(f"Reached final: {df['is_final_reached'].sum()}")

    low_score_advanced = df[df["low_score_advanced_count"] > 0]
    print(f"Low score but advanced (at least once): {len(low_score_advanced)}")

    print("\n=== Low Score Advanced Count Distribution ===")
    count_dist = (
        df[df["low_score_advanced_count"] > 0]["low_score_advanced_count"]
        .value_counts()
        .sort_index()
    )
    for count, num in count_dist.items():
        print(f"  {count} time(s): {num} contestants")

    print("\n=== Last Low Score Advanced Week ===")
    week_counts = (
        df[df["low_score_advanced_week"] > 0]["low_score_advanced_week"]
        .value_counts()
        .sort_index()
    )
    for week, count in week_counts.items():
        print(f"  Week {week}: {count} contestants")

    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
