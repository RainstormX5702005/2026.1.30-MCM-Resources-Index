import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, List

from configs.config import OUTPUT_DIR


def load_prediction_data(file_name: str) -> pd.DataFrame:
    """
    加载MCMC预测的观众投票比例数据

    Args:
        file_name: 预测数据文件名

    Returns:
        预测数据框
    """
    file_path = OUTPUT_DIR / "trained" / file_name
    df = pd.read_csv(file_path)
    return df


def load_percentage_data(file_name: str) -> pd.DataFrame:
    """
    加载基于百分比计算的原始数据

    Args:
        file_name: 原始数据文件名

    Returns:
        原始数据框
    """
    file_path = OUTPUT_DIR / file_name
    df = pd.read_csv(file_path)
    return df


def get_week_percentage(row: pd.Series, week: int) -> float:
    """
    获取某一行某一周的评委打分百分比

    Args:
        row: 数据行
        week: 周次

    Returns:
        该周的百分比（如果不存在则返回0）
    """
    col_name = f"week{week}_percentage"
    if col_name in row.index:
        value = row[col_name]
        if pd.notna(value) and value > 0:
            return value
    return 0.0


def get_elimination_week(result: str) -> int:
    """
    从results字段解析淘汰周次

    Args:
        result: results字段内容

    Returns:
        淘汰周次（如果不是被淘汰则返回-1）
    """
    if pd.isna(result):
        return -1

    # 匹配 "Eliminated Week X" 格式
    match = re.search(r"Eliminated Week (\d+)", str(result), re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 如果是获奖选手，返回-1
    if any(place in str(result) for place in ["1st", "2nd", "3rd", "Place"]):
        return -1

    return -1


def verify_elimination_prediction(
    pred_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    season_range: Tuple[int, int] = (3, 27),
) -> pd.DataFrame:
    """
    验证淘汰预测的准确性

    思路：
    1. 对于每个(season, week)，计算每个选手的总比例 = vote_mean + weekX_percentage
    2. 按总比例排名（最低的人最可能被淘汰）
    3. 检查实际被淘汰的人是否是排名最低的

    Args:
        pred_df: MCMC预测数据
        orig_df: 原始数据（包含weekX_percentage和results）
        season_range: 验证的赛季范围

    Returns:
        验证结果数据框
    """
    # 筛选指定赛季范围
    pred_filtered = pred_df[
        pred_df["season"].between(season_range[0], season_range[1])
    ].copy()

    orig_filtered = orig_df[
        orig_df["season"].between(season_range[0], season_range[1])
    ].copy()

    # 构建原始数据的查找字典
    # key: (season, celebrity_name) -> row
    orig_lookup = {}
    for _, row in orig_filtered.iterrows():
        key = (row["season"], row["celebrity_name"])
        orig_lookup[key] = row

    # 收集验证结果
    verification_results = []

    # 按(season, week)分组处理
    for (season, week), group in pred_filtered.groupby(["season", "week"]):
        # 计算每个选手的总比例
        contestants = []

        for _, pred_row in group.iterrows():
            celeb = pred_row["celebrity_name"]
            vote_mean = pred_row["vote_mean"]

            # 获取该选手的原始数据
            orig_key = (season, celeb)
            if orig_key not in orig_lookup:
                continue

            orig_row = orig_lookup[orig_key]

            # 获取该周的评委打分百分比
            judge_percentage = get_week_percentage(orig_row, week)

            # 如果该周没有评委打分，跳过（说明该选手当周没有参赛）
            if judge_percentage <= 0:
                continue

            # 计算总比例
            total_percentage = vote_mean + judge_percentage

            # 获取淘汰信息
            elim_week = get_elimination_week(orig_row["results"])
            was_eliminated_this_week = elim_week == week

            contestants.append(
                {
                    "season": season,
                    "week": week,
                    "celebrity_name": celeb,
                    "vote_mean": vote_mean,
                    "judge_percentage": judge_percentage,
                    "total_percentage": total_percentage,
                    "actual_elim_week": elim_week,
                    "was_eliminated_this_week": was_eliminated_this_week,
                    "results": orig_row["results"],
                }
            )

        if len(contestants) < 2:
            continue

        # 按总比例排名（最低的排名为1）
        contestants_df = pd.DataFrame(contestants)
        contestants_df["rank"] = (
            contestants_df["total_percentage"]
            .rank(method="min", ascending=True)
            .astype(int)
        )

        # 找出排名最低的选手（rank=1）
        lowest_ranked = contestants_df[contestants_df["rank"] == 1]

        # 找出实际被淘汰的选手
        actually_eliminated = contestants_df[
            contestants_df["was_eliminated_this_week"] == True
        ]

        # 判断预测是否正确
        prediction_correct = False
        eliminated_celeb = None
        eliminated_rank = None

        if len(actually_eliminated) > 0:
            eliminated_celeb = actually_eliminated.iloc[0]["celebrity_name"]
            eliminated_rank = actually_eliminated.iloc[0]["rank"]

            # 检查被淘汰的人是否在排名最低的组里
            if eliminated_celeb in lowest_ranked["celebrity_name"].values:
                prediction_correct = True

        # 记录验证结果
        for _, row in contestants_df.iterrows():
            row_dict = row.to_dict()
            row_dict["lowest_in_week"] = row["rank"] == 1
            row_dict["prediction_correct"] = prediction_correct
            row_dict["eliminated_this_week"] = eliminated_celeb
            row_dict["eliminated_rank"] = eliminated_rank
            row_dict["total_contestants"] = len(contestants_df)
            verification_results.append(row_dict)

    return pd.DataFrame(verification_results)


def calculate_accuracy_metrics(verify_df: pd.DataFrame) -> Dict:
    """
    计算验证准确率指标

    Args:
        verify_df: 验证结果数据框

    Returns:
        准确率指标字典
    """
    # 只保留有淘汰事件的周次
    weeks_with_elimination = verify_df[
        verify_df["eliminated_this_week"].notna()
    ].drop_duplicates(subset=["season", "week"])

    total_elimination_events = len(weeks_with_elimination)
    correct_predictions = weeks_with_elimination["prediction_correct"].sum()

    # 计算不同宽松度的准确率
    # 严格准确率：被淘汰者排名第1
    strict_correct = (weeks_with_elimination["eliminated_rank"] == 1).sum()

    # 宽松准确率：被淘汰者排名在倒数2名内
    lenient_correct = (weeks_with_elimination["eliminated_rank"] <= 2).sum()

    # 更宽松：被淘汰者排名在倒数3名内
    more_lenient_correct = (weeks_with_elimination["eliminated_rank"] <= 3).sum()

    return {
        "total_elimination_events": total_elimination_events,
        "strict_correct_count": strict_correct,
        "strict_accuracy": (
            strict_correct / total_elimination_events
            if total_elimination_events > 0
            else 0
        ),
        "lenient_correct_count": lenient_correct,
        "lenient_accuracy": (
            lenient_correct / total_elimination_events
            if total_elimination_events > 0
            else 0
        ),
        "more_lenient_correct_count": more_lenient_correct,
        "more_lenient_accuracy": (
            more_lenient_correct / total_elimination_events
            if total_elimination_events > 0
            else 0
        ),
    }


def print_detailed_analysis(verify_df: pd.DataFrame) -> None:
    """
    打印详细的分析结果

    Args:
        verify_df: 验证结果数据框
    """
    print("\n" + "=" * 80)
    print("详细淘汰预测分析")
    print("=" * 80)

    # 只看有淘汰事件的周次
    weeks_with_elimination = verify_df[
        verify_df["was_eliminated_this_week"] == True
    ].copy()

    # 按season, week排序
    weeks_with_elimination = weeks_with_elimination.sort_values(["season", "week"])

    for _, row in weeks_with_elimination.iterrows():
        season = row["season"]
        week = row["week"]
        celeb = row["celebrity_name"]
        rank = row["rank"]
        total = row["total_contestants"]
        vote_mean = row["vote_mean"]
        judge_pct = row["judge_percentage"]
        total_pct = row["total_percentage"]

        # 判断预测结果
        if rank == 1:
            status = "✅ 预测正确"
        elif rank <= 2:
            status = "⚠️ 排名倒数第2"
        elif rank <= 3:
            status = "⚠️ 排名倒数第3"
        else:
            status = "❌ 预测失败"

        print(f"\nSeason {season}, Week {week}: {celeb}")
        print(f"  观众投票比例: {vote_mean:.4f}")
        print(f"  评委打分比例: {judge_pct:.4f}")
        print(f"  总比例: {total_pct:.4f}")
        print(f"  排名: {rank}/{total} (1=最低)")
        print(f"  {status}")


def main():
    """主函数：执行验证流程"""

    print("=" * 80)
    print("MCMC观众投票预测验证")
    print("验证范围: Season 3 - Season 27 (使用percentage计算的数据)")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    pred_df = load_prediction_data("weekly_audience_vote_share_with_95CI_STABLE.csv")
    orig_df = load_percentage_data("processed_data_percentage.csv")

    print(f"      预测数据: {len(pred_df)} 条记录")
    print(f"      原始数据: {len(orig_df)} 条记录")

    # 检查season范围
    pred_seasons = pred_df["season"].unique()
    orig_seasons = orig_df["season"].unique()
    print(f"      预测数据季节范围: {pred_seasons.min()} - {pred_seasons.max()}")
    print(f"      原始数据季节范围: {orig_seasons.min()} - {orig_seasons.max()}")

    # 2. 执行验证
    print("\n[2/4] 执行验证...")
    # 注意：processed_data_percentage.csv 包含的是 season 1-2 和 28-33
    # 因为 data_sparse 函数把 3-27 放到了 rank 数据集
    # 所以我们需要验证原始数据实际包含的季节
    actual_seasons = orig_df["season"].unique()
    print(f"      实际季节: {sorted(actual_seasons)}")

    # 根据实际数据调整验证范围
    min_season = int(actual_seasons.min())
    max_season = int(actual_seasons.max())

    verify_df = verify_elimination_prediction(
        pred_df, orig_df, season_range=(min_season, max_season)
    )

    print(f"      验证记录: {len(verify_df)} 条")

    # 3. 计算准确率
    print("\n[3/4] 计算准确率指标...")
    metrics = calculate_accuracy_metrics(verify_df)

    print(f"\n      总淘汰事件数: {metrics['total_elimination_events']}")
    print(f"\n      严格准确率 (排名最低被淘汰):")
    print(
        f"        正确预测: {metrics['strict_correct_count']}/{metrics['total_elimination_events']}"
    )
    print(f"        准确率: {metrics['strict_accuracy']:.2%}")
    print(f"\n      宽松准确率 (排名倒数2内被淘汰):")
    print(
        f"        正确预测: {metrics['lenient_correct_count']}/{metrics['total_elimination_events']}"
    )
    print(f"        准确率: {metrics['lenient_accuracy']:.2%}")
    print(f"\n      更宽松准确率 (排名倒数3内被淘汰):")
    print(
        f"        正确预测: {metrics['more_lenient_correct_count']}/{metrics['total_elimination_events']}"
    )
    print(f"        准确率: {metrics['more_lenient_accuracy']:.2%}")

    # 4. 详细分析
    print("\n[4/4] 详细分析...")
    print_detailed_analysis(verify_df)

    # 5. 保存验证结果
    output_path = OUTPUT_DIR / "verified" / "verification_results.csv"
    verify_df.to_csv(output_path, index=False)
    print(f"\n✅ 验证结果已保存到: {output_path}")

    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
