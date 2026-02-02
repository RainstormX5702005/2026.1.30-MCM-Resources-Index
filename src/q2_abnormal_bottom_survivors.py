"""
寻找评分末两位但仍晋级的异常选手 (v2版本)

使用 q4_featured_data.csv 数据源，与 q2_wilcoxon_v2 保持一致
逻辑不变：识别某周评委分数排名倒数第1或第2但仍晋级的选手
"""

import pandas as pd
import numpy as np

from configs.config import OUTPUT_DIR


def load_data() -> pd.DataFrame:
    """加载 q4_featured_data.csv"""
    df = pd.read_csv(
        OUTPUT_DIR / "q4_featured_data.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    print(f"  ✓ 加载数据: {len(df)} 行, {df['season'].nunique()} 个赛季")
    return df


def transform_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    将宽格式数据转换为长格式

    每行代表一个选手在某季某周的数据
    """
    records = []

    for _, row in df.iterrows():
        # 计算参与周数
        weeks_participated = 0
        for week in range(1, 12):
            score_col = f"week{week}_score_sum"
            if (
                score_col in df.columns
                and pd.notna(row[score_col])
                and row[score_col] > 0
            ):
                weeks_participated = week

        # 计算 placement（实际名次）
        season_total = row.get("season_total_contestants", 12)
        relative_placement = row.get("relative_placement", 0)
        placement = int(round(relative_placement * (season_total - 1))) + 1

        for week in range(1, 12):
            score_col = f"week{week}_score_sum"

            if score_col not in df.columns:
                continue

            score = row[score_col]

            # 跳过无效数据
            if pd.isna(score) or score <= 0:
                continue

            records.append(
                {
                    "celebrity_name": row["celebrity_name"],
                    "season": row["season"],
                    "week": week,
                    "judge_score": score,
                    "placement": placement,
                    "weeks_participated": weeks_participated,
                }
            )

    long_df = pd.DataFrame(records)
    print(f"  ✓ 转换为长格式: {len(long_df)} 行")
    return long_df


def identify_bottom_survivors(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别排在末两位但仍然晋级的异常选手

    逻辑：
    1. 在某周排名倒数第1或第2（基于评委分数）
    2. 但该周不是该选手的最后一周（即该选手继续参加了下一周）
    """
    df = df.copy()

    # 计算每周的评委分数排名（分数越低，排名越差）
    df["judge_rank"] = df.groupby(["season", "week"])["judge_score"].rank(
        ascending=True, method="min"
    )

    # 每周参赛人数
    df["n_contestants"] = df.groupby(["season", "week"])["celebrity_name"].transform(
        "count"
    )

    # 判断是否排在倒数两位
    df["is_bottom_two"] = (df["judge_rank"] <= 2).astype(int)

    # 判断该周是否是该选手的最后一周
    df["is_last_week"] = (df["week"] == df["weeks_participated"]).astype(int)

    # 异常情况：排在倒数两位，但不是最后一周（即晋级了）
    df["is_abnormal"] = (df["is_bottom_two"] == 1) & (df["is_last_week"] == 0)

    return df


def main():
    """寻找评分末两位但仍晋级的异常选手"""
    print("=" * 80)
    print("寻找评分末两位但仍晋级的异常选手 (v2)")
    print("=" * 80)
    print("\n定义:")
    print("  异常 = 某周评委分数排名倒数第1或第2，但该选手仍继续参加下一周")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    df = load_data()

    # 2. 转换为长格式
    print("\n[2/3] 转换为长格式...")
    long_df = transform_to_long_format(df)
    print(
        f"  ✓ {long_df['celebrity_name'].nunique()} 个选手, {long_df['season'].nunique()} 个赛季"
    )

    # 3. 识别异常选手
    print("\n[3/3] 识别异常情况...")
    result_df = identify_bottom_survivors(long_df)

    # 筛选出异常情况
    abnormal_df = result_df[result_df["is_abnormal"] == True].copy()
    abnormal_df = abnormal_df.sort_values(["season", "week", "judge_rank"])

    print(f"  ✓ 发现 {len(abnormal_df)} 个异常事件")

    # 统计信息
    print("\n" + "=" * 80)
    print("异常情况统计")
    print("=" * 80)

    print(f"\n总异常事件数: {len(abnormal_df)}")
    print(f"涉及选手数: {abnormal_df['celebrity_name'].nunique()}")
    print(f"涉及赛季数: {abnormal_df['season'].nunique()}")

    # 按排名统计
    print("\n按排名分布:")
    rank_counts = abnormal_df["judge_rank"].value_counts().sort_index()
    for rank, count in rank_counts.items():
        print(f"  倒数第{int(rank)}名: {count} 次")

    # 按赛季统计
    print("\n按赛季分布 (前10):")
    season_counts = abnormal_df["season"].value_counts().sort_index().head(10)
    for season, count in season_counts.items():
        print(f"  Season {int(season)}: {count} 次异常")

    # 按周统计
    print("\n按周分布:")
    week_counts = abnormal_df["week"].value_counts().sort_index()
    for week, count in week_counts.items():
        print(f"  Week {int(week)}: {count} 次异常")

    # 显示一些具体案例
    print("\n" + "=" * 80)
    print("异常案例示例 (前20个):")
    print("=" * 80)

    display_cols = [
        "season",
        "week",
        "celebrity_name",
        "judge_score",
        "judge_rank",
        "n_contestants",
        "weeks_participated",
        "placement",
    ]

    print(abnormal_df[display_cols].head(20).to_string(index=False))

    # 找出最极端的案例（排名倒数第1但晋级）
    print("\n" + "=" * 80)
    print("最极端案例 (倒数第1名仍晋级):")
    print("=" * 80)

    extreme_cases = abnormal_df[abnormal_df["judge_rank"] == 1]
    print(f"\n发现 {len(extreme_cases)} 个倒数第1名仍晋级的案例")

    if len(extreme_cases) > 0:
        print("\n案例详情:")
        print(extreme_cases[display_cols].head(20).to_string(index=False))

        # 统计最常出现的极端案例选手
        print("\n最常出现倒数第1名仍晋级的选手 (Top 10):")
        extreme_contestants = (
            extreme_cases.groupby("celebrity_name")
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        for name, count in extreme_contestants.items():
            season = extreme_cases[extreme_cases["celebrity_name"] == name][
                "season"
            ].iloc[0]
            placement = extreme_cases[extreme_cases["celebrity_name"] == name][
                "placement"
            ].iloc[0]
            print(f"  {name} (Season {int(season)}, 最终第{placement}名): {count} 次")

    # 保存结果
    output_path = OUTPUT_DIR / "question2_res" / "abnormal"
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存完整的结果数据
    result_df.to_csv(
        output_path / "bottom_survivors_full.csv", index=False, encoding="utf-8"
    )

    # 保存异常情况
    abnormal_df.to_csv(
        output_path / "bottom_survivors_abnormal.csv", index=False, encoding="utf-8"
    )

    # 创建汇总统计
    summary = {
        "total_abnormal_events": len(abnormal_df),
        "unique_contestants": abnormal_df["celebrity_name"].nunique(),
        "unique_seasons": abnormal_df["season"].nunique(),
        "rank_1_survivors": len(abnormal_df[abnormal_df["judge_rank"] == 1]),
        "rank_2_survivors": len(abnormal_df[abnormal_df["judge_rank"] == 2]),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        output_path / "bottom_survivors_v2_summary.csv", index=False, encoding="utf-8"
    )

    print(f"\n✓ 结果已保存:")
    print(f"  完整数据: {output_path / 'bottom_survivors_v2_full.csv'}")
    print(f"  异常数据: {output_path / 'bottom_survivors_v2_abnormal.csv'}")
    print(f"  汇总统计: {output_path / 'bottom_survivors_v2_summary.csv'}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
