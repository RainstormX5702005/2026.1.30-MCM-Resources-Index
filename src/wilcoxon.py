import pandas as pd
import numpy as np
import re

from scipy.stats import wilcoxon
from typing import Tuple

from configs.config import OUTPUT_DIR


def load_processed_data(
    processed_file: str, votes_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载已处理的数据和投票数据

    Args:
        processed_file: 已处理的数据文件名
        votes_file: 投票数据文件名

    Returns:
        (processed_df, votes_df)
    """
    processed_path = OUTPUT_DIR / processed_file
    votes_path = OUTPUT_DIR / "trained" / votes_file

    processed = pd.read_csv(processed_path, sep=",", header=0, encoding="utf-8")
    votes = pd.read_csv(votes_path, sep=",", header=0, encoding="utf-8")

    # 标准化列名和数据类型
    processed["contestant"] = processed["celebrity_name"].astype(str).str.strip()
    votes["contestant"] = votes["celebrity_name"].astype(str).str.strip()

    processed["season"] = pd.to_numeric(processed["season"], errors="coerce")
    votes["season"] = pd.to_numeric(votes["season"], errors="coerce")
    votes["week"] = pd.to_numeric(votes["week"], errors="coerce")
    votes["vote_mean"] = pd.to_numeric(votes["vote_mean"], errors="coerce")

    # 提取淘汰周
    processed["results"] = processed["results"].astype(str)
    processed["eliminated_week"] = processed["results"].str.extract(
        r"Eliminated\s*Week\s*(\d+)", expand=False
    )
    processed["eliminated_week"] = pd.to_numeric(
        processed["eliminated_week"], errors="coerce"
    )

    return processed, votes


def transform_to_weekly_format(
    processed_df: pd.DataFrame, votes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    将宽格式数据转换为每周长格式，并合并投票数据

    Args:
        processed_df: 已处理的宽格式数据（包含 week{N}_score_sum）
        votes_df: 投票数据

    Returns:
        包含 season, week, contestant, judge_total, vote_mean, eliminated_actual 的 DataFrame
    """
    # 找到所有 week_score_sum 列
    week_score_sum_cols = [
        c for c in processed_df.columns if re.match(r"^week\d+_score_sum$", str(c))
    ]

    if not week_score_sum_cols:
        raise ValueError("未找到 week{N}_score_sum 列，请检查输入数据")

    # 宽格式 -> 长格式
    long_df = processed_df.melt(
        id_vars=["season", "contestant", "eliminated_week"],
        value_vars=week_score_sum_cols,
        var_name="week_col",
        value_name="judge_total",
    )

    long_df["week"] = (
        long_df["week_col"].str.extract(r"week(\d+)", expand=False).astype(int)
    )
    long_df = long_df.drop(columns=["week_col"])
    long_df = long_df[long_df["judge_total"] > 0].copy()
    long_df["eliminated_actual"] = (
        (long_df["week"] == long_df["eliminated_week"]).fillna(False).astype(int)
    )

    merged_df = long_df.merge(
        votes_df[["season", "week", "contestant", "vote_mean"]],
        on=["season", "week", "contestant"],
        how="left",
    )

    merged_df = merged_df.dropna(subset=["judge_total", "vote_mean"]).copy()

    return merged_df


def calculate_share_method_positions(
    df: pd.DataFrame, w_judge: float = 0.5
) -> pd.DataFrame:
    """
    计算百分比法的综合指标和位置排名

    Args:
        df: 包含 judge_total 和 vote_mean 的 DataFrame
        w_judge: 评委分数权重（默认 0.5）

    Returns:
        添加 judge_share, fan_share, combined_share, pos_share 列的 DataFrame
    """
    df = df.copy()

    df["judge_share"] = df["judge_total"] / df.groupby(["season", "week"])[
        "judge_total"
    ].transform("sum")

    df["fan_share"] = df["vote_mean"]
    # highlight 使用 linear_interpolation 进行特征的合成以便于检验
    df["combined_share"] = w_judge * df["judge_share"] + (1 - w_judge) * df["fan_share"]
    df["pos_share"] = df.groupby(["season", "week"])["combined_share"].rank(
        ascending=False, method="average"
    )

    return df


def calculate_rank_method_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算排名法的综合指标和位置排名

    Args:
        df: 包含 judge_total 和 vote_mean 的 DataFrame

    Returns:
        添加 judge_rank, fan_rank, rank_sum, pos_rank 列的 DataFrame
    """
    df = df.copy()

    df["judge_rank"] = df.groupby(["season", "week"])["judge_total"].rank(
        ascending=False, method="average"
    )
    df["fan_rank"] = df.groupby(["season", "week"])["vote_mean"].rank(
        ascending=False, method="average"
    )
    df["rank_sum"] = df["judge_rank"] + df["fan_rank"]
    df["pos_rank"] = df.groupby(["season", "week"])["rank_sum"].rank(
        ascending=True, method="average"
    )

    return df


def extract_elimination_positions(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    提取每周真实淘汰者的位置信息

    Args:
        weekly_df: 包含所有选手每周位置的 DataFrame

    Returns:
        仅包含淘汰者的 DataFrame，添加距离最差位置的指标
    """
    elim_df = weekly_df[weekly_df["eliminated_actual"] == 1].copy()

    # 处理双淘汰情况：每周保留第一个淘汰者
    elim_df = (
        elim_df.sort_values(["season", "week", "contestant"])
        .groupby(["season", "week"], as_index=False)
        .head(1)
    )

    n_week_map = (
        weekly_df.groupby(["season", "week"])["contestant"]
        .count()
        .rename("n_week")
        .reset_index()
    )
    elim_df = elim_df.merge(n_week_map, on=["season", "week"], how="left")

    elim_df["dist_to_worst_share"] = elim_df["n_week"] - elim_df["pos_share"]
    elim_df["dist_to_worst_rank"] = elim_df["n_week"] - elim_df["pos_rank"]

    return elim_df


def perform_wilcoxon_test(elim_df: pd.DataFrame) -> Tuple[float, float, int]:
    """
    对百分比法和排名法的淘汰者位置进行配对 Wilcoxon 检验

    Args:
        elim_df: 淘汰者位置数据

    Returns:
        (统计量, p值, 样本数)
    """
    x = elim_df["dist_to_worst_share"].to_numpy()
    y = elim_df["dist_to_worst_rank"].to_numpy()

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 1:
        raise ValueError("没有足够的有效数据进行 Wilcoxon 检验")

    stat, p = wilcoxon(x - y, alternative="two-sided", zero_method="wilcox")

    return stat, p, len(x)


def main():
    """执行 Wilcoxon 符号秩检验分析"""
    # 加载数据
    processed_df, votes_df = load_processed_data(
        "processed_data.csv", "weekly_audience_vote_share_with_95CI_STABLE.csv"
    )

    # 转换为每周格式并合并投票数据
    weekly_df = transform_to_weekly_format(processed_df, votes_df)

    weekly_df = calculate_share_method_positions(weekly_df, w_judge=0.5)
    weekly_df = calculate_rank_method_positions(weekly_df)

    elim_df = extract_elimination_positions(weekly_df)

    stat, p_value, n = perform_wilcoxon_test(elim_df)

    output_path = OUTPUT_DIR / "question2_res" / "wilcoxon_test_result.csv"
    weekly_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✓ Wilcoxon 符号秩检验完成")
    print(f"  样本数: {n}")
    print(f"  统计量: {stat:.4f}")
    print(f"  P值: {p_value:.4f}")
    print(f"  结果文件: {output_path}")


def run_with_modeling_data(data_type: str = "rank"):
    """
    使用 modeling_results_detailed 转换后的数据运行 Wilcoxon 检验

    Args:
        data_type: "rank" 或 "pct"
    """
    processed_file = f"processed_modeling_{data_type}.csv"
    votes_file = f"modeling_votes_{data_type}.csv"

    processed_df, votes_df = load_processed_data(processed_file, votes_file)

    weekly_df = transform_to_weekly_format(processed_df, votes_df)
    weekly_df = calculate_share_method_positions(weekly_df, w_judge=0.5)
    weekly_df = calculate_rank_method_positions(weekly_df)

    elim_df = extract_elimination_positions(weekly_df)

    stat, p_value, n = perform_wilcoxon_test(elim_df)

    output_path = OUTPUT_DIR / "question2_res" / f"wilcoxon_modeling_{data_type}.csv"
    weekly_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✓ Wilcoxon 检验完成 (modeling_{data_type})")
    print(f"  样本数: {n}")
    print(f"  统计量: {stat:.4f}")
    print(f"  P值: {p_value:.4f}")
    print(f"  结果文件: {output_path}")

    return stat, p_value, n


if __name__ == "__main__":
    main()
