# -*- coding: utf-8 -*-
"""
分析争议选手在倒数两位的情况 - 基于 q4_featured_data.csv
生成适合绘制电泳图的CSV文件
"""

import pandas as pd
import numpy as np
from configs.config import OUTPUT_DIR


def main():
    print("\n" + "=" * 80)
    print("争议选手倒数两位分析 (基于 q4_featured_data.csv)")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    featured_df = pd.read_csv(OUTPUT_DIR / "q4_featured_data.csv")
    processed_df = pd.read_csv(OUTPUT_DIR / "processed_data.csv")

    # 合并placement、weeks_participated和low_score_advanced_count
    featured_df = featured_df.merge(
        processed_df[
            [
                "celebrity_name",
                "season",
                "placement",
                "weeks_participated",
                "low_score_advanced_count",
            ]
        ],
        on=["celebrity_name", "season"],
        how="left",
    )
    print(f"  加载 {len(featured_df)} 个选手记录")

    # 2. 识别争议选手（使用 low_score_advanced_count >= 1）
    print("\n[2/6] 识别争议选手（使用 low_score_advanced_count >= 1）...")

    # 使用 low_score_advanced_count 字段识别争议选手
    mask = processed_df["low_score_advanced_count"] >= 1
    controversial_df = processed_df[mask][
        ["celebrity_name", "season", "low_score_advanced_count"]
    ].copy()

    controversial = list(
        controversial_df[["celebrity_name", "season"]].itertuples(
            index=False, name=None
        )
    )

    print(f"  识别 {len(controversial)} 个争议选手")
    print(f"  前20个争议选手:")
    for i, row in controversial_df.head(20).iterrows():
        print(
            f"    {row['celebrity_name']} (Season {int(row['season'])}) - 争议次数: {int(row['low_score_advanced_count'])}"
        )

    # 3. 转换为长格式
    print("\n[3/6] 转换为长格式...")
    records = []
    for _, row in featured_df.iterrows():
        for week in range(1, 12):
            score_col = f"week{week}_score_sum"
            fan_col = f"week{week}_audience_votes"
            rank_col = f"week{week}_combined_rank"
            pct_col = f"week{week}_combined_pct"

            score = row.get(score_col, -1)
            fan = row.get(fan_col, -1)
            rank = row.get(rank_col, -1)
            pct = row.get(pct_col, -1)

            if pd.isna(score) or score <= 0:
                continue
            if pd.isna(fan) or fan <= 0:
                continue
            if pd.isna(rank) or rank <= 0:
                continue
            if pd.isna(pct) or pct <= 0:
                continue

            records.append(
                {
                    "celebrity_name": row["celebrity_name"],
                    "season": row["season"],
                    "week": week,
                    "judge_score": score,
                    "fan_votes": fan,
                    "method1_rank": rank,
                    "method2_pct": pct,
                    "placement": row["placement"],
                    "weeks_participated": row["weeks_participated"],
                }
            )

    long_df = pd.DataFrame(records)
    print(f"  生成 {len(long_df)} 条记录")

    # 4. 计算位置排名
    print("\n[4/6] 计算位置排名...")
    long_df["method1_pos"] = long_df.groupby(["season", "week"])["method1_rank"].rank(
        ascending=True, method="average"
    )
    long_df["method2_pos"] = long_df.groupby(["season", "week"])["method2_pct"].rank(
        ascending=False, method="average"
    )

    # 识别倒数两位
    long_df["n_contestants"] = long_df.groupby(["season", "week"])[
        "celebrity_name"
    ].transform("count")
    long_df["method1_is_bottom_two"] = (
        long_df["method1_pos"] >= long_df["n_contestants"] - 1
    ).astype(int)
    long_df["method2_is_bottom_two"] = (
        long_df["method2_pos"] >= long_df["n_contestants"] - 1
    ).astype(int)
    long_df["both_methods_bottom_two"] = (
        (long_df["method1_is_bottom_two"] == 1)
        & (long_df["method2_is_bottom_two"] == 1)
    ).astype(int)

    print(f"  计算完成")

    # 5. 筛选争议选手
    print("\n[5/6] 筛选争议选手的倒数两位情况...")
    controversial_set = set(controversial)
    controversial_data = long_df[
        long_df.apply(
            lambda x: (x["celebrity_name"], x["season"]) in controversial_set, axis=1
        )
    ].copy()

    bottom_cases = controversial_data[
        (controversial_data["method1_is_bottom_two"] == 1)
        | (controversial_data["method2_is_bottom_two"] == 1)
    ].copy()

    both_bottom = controversial_data[
        controversial_data["both_methods_bottom_two"] == 1
    ].copy()

    print(f"  争议选手数据: {len(controversial_data)} 条")
    print(f"  倒数两位情况: {len(bottom_cases)} 条")
    print(f"  两种方法都倒数: {len(both_bottom)} 条")

    # 6. 标记提前淘汰风险
    print("\n[6/6] 标记提前淘汰风险...")
    result_records = []

    for _, row in both_bottom.iterrows():
        season = row["season"]
        week = row["week"]
        name = row["celebrity_name"]
        score = row["judge_score"]
        weeks_participated = row["weeks_participated"]

        # 跳过最后一周
        if week == weeks_participated:
            continue

        # 找同周倒数两名的选手
        week_data = long_df[
            (long_df["season"] == season) & (long_df["week"] == week)
        ].copy()
        week_data = week_data.sort_values("judge_score", ascending=True)
        bottom_two = week_data.head(2)

        if len(bottom_two) < 2:
            continue

        companion = bottom_two[bottom_two["celebrity_name"] != name]
        if len(companion) == 0:
            continue

        companion_row = companion.iloc[0]
        companion_name = companion_row["celebrity_name"]
        companion_score = companion_row["judge_score"]

        is_eliminated_early = 1 if score < companion_score else 0

        row_dict = row.to_dict()
        row_dict["companion_name"] = companion_name
        row_dict["companion_judge_score"] = companion_score
        row_dict["score_difference"] = score - companion_score
        row_dict["is_eliminated_early"] = is_eliminated_early

        result_records.append(row_dict)

    both_bottom_with_risk = pd.DataFrame(result_records)
    print(f"  有效记录: {len(both_bottom_with_risk)} 条")

    if len(both_bottom_with_risk) > 0:
        risk_count = both_bottom_with_risk["is_eliminated_early"].sum()
        print(f"  本应被提前淘汰: {risk_count} 次")

    # 7. 保存结果
    print("\n[7/7] 保存结果...")
    output_path = OUTPUT_DIR / "question2_res" / "q4"
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存所有倒数两位情况
    bottom_cases.to_csv(
        output_path / "controversial_bottom_two_all.csv", index=False, encoding="utf-8"
    )

    # 保存两种方法都倒数的情况（带提前淘汰标记）
    both_bottom_with_risk.to_csv(
        output_path / "controversial_both_methods_bottom.csv",
        index=False,
        encoding="utf-8",
    )

    # 电泳图数据（简化版）
    if len(both_bottom_with_risk) > 0:
        electrophoresis_cols = [
            "season",
            "week",
            "celebrity_name",
            "weeks_participated",
            "placement",
            "judge_score",
            "companion_name",
            "companion_judge_score",
            "score_difference",
            "is_eliminated_early",
        ]
        both_bottom_with_risk[electrophoresis_cols].to_csv(
            output_path / "controversial_both_bottom_for_electrophoresis.csv",
            index=False,
            encoding="utf-8",
        )

    print("\n结果已保存:")
    print(f"  1. {output_path / 'controversial_bottom_two_all.csv'}")
    print(f"  2. {output_path / 'controversial_both_methods_bottom.csv'}")
    print(f"  3. {output_path / 'controversial_both_bottom_for_electrophoresis.csv'}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
