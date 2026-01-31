import pandas as pd
import numpy as np

from scipy.stats import wilcoxon
from typing import Tuple

from configs.config import DATA_DIR, OUTPUT_DIR


def load_main_data() -> pd.DataFrame:
    """加载主数据集 processed_data.csv"""
    df = pd.read_csv(
        OUTPUT_DIR / "processed_data.csv", sep=",", header=0, encoding="utf-8"
    )
    return df


def load_fan_votes_data() -> pd.DataFrame:
    """
    加载粉丝投票预测数据

    - Season 1, 2, 28-34: 使用 rank 数据集
    - Season 3-27: 使用 pct 数据集

    注意：预测数据中的 celebrity_name 实际上是 contestant_id
    需要通过 (season, week, placement, weeks_participated) 来关联
    """
    # 加载 rank 数据 (season 1, 2, 28-34)
    rank_df = pd.read_csv(
        DATA_DIR / "modeling_results_detailed_rank.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    rank_df["week"] = rank_df["week_idx"] + 1
    rank_df["data_source"] = "rank"

    # 加载 pct 数据 (season 3-27)
    pct_df = pd.read_csv(
        DATA_DIR / "modeling_results_detailed_pc.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    pct_df["season"] = pct_df["season_idx"] + 3
    pct_df["week"] = pct_df["week_idx"] + 1
    pct_df["data_source"] = "pct"
    # pct 数据用 judge_score 作为 score_sum
    pct_df["score_sum"] = pct_df["judge_score"]

    # 选择共同的列（不包括 celebrity_name，因为那是 contestant_id）
    common_cols = [
        "season",
        "week",
        "fan_votes",
        "placement",
        "weeks_participated",
        "data_source",
    ]

    # 合并两个数据集
    combined = pd.concat([rank_df[common_cols], pct_df[common_cols]], ignore_index=True)

    return combined


def transform_to_long_format(main_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 processed_data 的宽格式转换为长格式

    每行代表一个选手在某季某周的数据
    """
    # 提取选手基本信息
    id_cols = ["celebrity_name", "season", "placement", "weeks_participated", "results"]

    # 提取每周评委分数
    score_cols = [col for col in main_df.columns if col.endswith("_score_sum")]

    # 转换为长格式
    records = []
    for _, row in main_df.iterrows():
        for week in range(1, 12):  # week 1-11
            score_col = f"week{week}_score_sum"
            if score_col in main_df.columns:
                score = row[score_col]
                if pd.notna(score) and score > 0:
                    records.append(
                        {
                            "celebrity_name": row["celebrity_name"],
                            "season": row["season"],
                            "week": week,
                            "judge_score": score,
                            "placement": row["placement"],
                            "weeks_participated": row["weeks_participated"],
                            "results": row["results"],
                        }
                    )

    long_df = pd.DataFrame(records)
    return long_df


def merge_fan_votes(long_df: pd.DataFrame, fan_votes_df: pd.DataFrame) -> pd.DataFrame:
    """
    将预测的粉丝投票数据合并到主数据集

    通过 (season, week, placement, weeks_participated) 来关联
    因为预测数据中的 celebrity_name 是 contestant_id
    """
    # 按 season, week, placement, weeks_participated 合并
    merged = long_df.merge(
        fan_votes_df[
            [
                "season",
                "week",
                "placement",
                "weeks_participated",
                "fan_votes",
                "data_source",
            ]
        ],
        on=["season", "week", "placement", "weeks_participated"],
        how="left",
    )

    # 检查合并情况
    missing = merged["fan_votes"].isna().sum()
    total = len(merged)
    matched = total - missing
    if missing > 0:
        print(
            f"  警告: {missing}/{total} 行缺少粉丝投票数据 (匹配率: {matched/total*100:.1f}%)"
        )
    else:
        print(f"  ✓ 全部 {total} 行成功匹配粉丝投票数据")

    return merged


def calculate_method_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算两种方法的位置排名

    方法1 (排名法): 对粉丝票数做逆序排名，排名越大越可能出局
    方法2 (占比法): 计算粉丝票数占比 + 评委票数占比，和越小越容易出局
    """
    df = df.copy()

    # ===== 方法1: 排名法 =====
    # 对粉丝票数做逆序排名（票数越少，排名越大，越可能出局）
    df["fan_vote_rank"] = df.groupby(["season", "week"])["fan_votes"].rank(
        ascending=False, method="average"
    )
    # 对评委分数也做逆序排名
    df["judge_score_rank"] = df.groupby(["season", "week"])["judge_score"].rank(
        ascending=False, method="average"
    )
    # 方法1: 排名之和
    df["method1_total_rank"] = df["fan_vote_rank"] + df["judge_score_rank"]
    # 位置排名：总排名越大，越靠后
    df["method1_pos"] = df.groupby(["season", "week"])["method1_total_rank"].rank(
        ascending=True, method="average"
    )

    # ===== 方法2: 占比法 =====
    # 计算评委分数占比
    df["judge_share"] = df["judge_score"] / df.groupby(["season", "week"])[
        "judge_score"
    ].transform("sum")

    # 计算粉丝投票占比
    df["fan_share"] = df["fan_votes"] / df.groupby(["season", "week"])[
        "fan_votes"
    ].transform("sum")

    # 方法2: 占比之和
    df["method2_total_share"] = df["judge_share"] + df["fan_share"]
    # 位置排名：占比越大越好，降序排名
    df["method2_pos"] = df.groupby(["season", "week"])["method2_total_share"].rank(
        ascending=False, method="average"
    )

    return df


def identify_eliminated(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别淘汰者

    淘汰周 = weeks_participated（最后参与的周）
    前3名不算淘汰
    """
    df = df.copy()

    # 淘汰周就是最后参与的周（对于 placement > 3 的选手）
    df["eliminated_week"] = df.apply(
        lambda row: row["weeks_participated"] if row["placement"] > 3 else np.nan,
        axis=1,
    )
    # 标记当前周是否为淘汰周
    df["is_eliminated"] = (df["week"] == df["eliminated_week"]).astype(int)

    return df


def extract_elimination_data(df: pd.DataFrame) -> pd.DataFrame:
    """提取淘汰者数据"""
    elim_df = df[df["is_eliminated"] == 1].copy()

    # 每周只取一个淘汰者（如果有多个）
    elim_df = (
        elim_df.sort_values(["season", "week", "celebrity_name"])
        .groupby(["season", "week"], as_index=False)
        .head(1)
    )

    # 每周参赛人数
    n_week_map = (
        df.groupby(["season", "week"])["celebrity_name"]
        .count()
        .rename("n_week")
        .reset_index()
    )
    elim_df = elim_df.merge(n_week_map, on=["season", "week"], how="left")

    # 计算距离最差位置的距离（0 = 最差位置，值越小 = 预测越准确）
    elim_df["dist_method1"] = elim_df["n_week"] - elim_df["method1_pos"]
    elim_df["dist_method2"] = elim_df["n_week"] - elim_df["method2_pos"]

    return elim_df


def perform_wilcoxon_test(elim_df: pd.DataFrame) -> Tuple[float, float, int]:
    """
    执行配对 Wilcoxon 检验
    """
    x = elim_df["dist_method1"].to_numpy()
    y = elim_df["dist_method2"].to_numpy()

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 1:
        raise ValueError("没有足够的有效数据进行 Wilcoxon 检验")

    stat, p = wilcoxon(x - y, alternative="two-sided", zero_method="wilcox")

    return stat, p, len(x)


def main():
    """执行 Wilcoxon 检验比较两种评估方法"""
    print("=" * 60)
    print("基于 processed_data 比较两种评估方法")
    print("=" * 60)
    print("\n数据来源:")
    print("  - 主数据集: processed_data.csv")
    print("  - Season 1, 2, 28-34: 使用 rank 数据集的粉丝票数")
    print("  - Season 3-27: 使用 pct 数据集的粉丝票数")
    print("\n方法1 (排名法): judge_rank + fan_vote_rank")
    print("  → 排名之和越大，越可能被淘汰")
    print("\n方法2 (占比法): judge_share + fan_share")
    print("  → 占比之和越小，越可能被淘汰")
    print("=" * 60)

    # 1. 加载主数据集
    print("\n[1/5] 加载主数据集...")
    main_df = load_main_data()
    print(f"  ✓ 加载 {len(main_df)} 个选手, {main_df['season'].nunique()} 个季度")

    # 2. 转换为长格式
    print("\n[2/5] 转换为长格式...")
    long_df = transform_to_long_format(main_df)
    print(f"  ✓ 生成 {len(long_df)} 条记录 (选手-周)")

    # 3. 加载并合并粉丝投票数据
    print("\n[3/5] 合并粉丝投票预测数据...")
    fan_votes_df = load_fan_votes_data()
    merged_df = merge_fan_votes(long_df, fan_votes_df)
    print(f"  ✓ 合并完成, {len(merged_df)} 条记录")

    # 过滤掉缺少粉丝投票的记录
    merged_df = merged_df.dropna(subset=["fan_votes"])
    print(f"  ✓ 有效记录: {len(merged_df)} 条")

    # 4. 计算两种方法的位置排名
    print("\n[4/5] 计算两种方法的位置排名...")
    result_df = calculate_method_positions(merged_df)
    result_df = identify_eliminated(result_df)
    print(f"  ✓ 计算完成")

    # 提取淘汰者数据
    elim_df = extract_elimination_data(result_df)
    print(f"  ✓ 识别 {len(elim_df)} 个淘汰事件")

    # 5. 执行 Wilcoxon 检验
    print("\n[5/5] 执行 Wilcoxon 配对符号秩检验...")
    stat, p_value, n = perform_wilcoxon_test(elim_df)

    # 保存结果
    output_path = OUTPUT_DIR / "question2_res"
    output_path.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(
        output_path / "wilcoxon_full_data.csv", index=False, encoding="utf-8"
    )
    elim_df.to_csv(
        output_path / "wilcoxon_eliminated.csv", index=False, encoding="utf-8"
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("Wilcoxon 配对符号秩检验结果")
    print("=" * 60)

    print(f"\n配对样本数: {n}")
    print(f"检验统计量: {stat:.4f}")
    print(f"P 值: {p_value:.6f}")
    print(
        f"显著性: {'显著差异 (p < 0.05)' if p_value < 0.05 else '无显著差异 (p >= 0.05)'}"
    )

    # 计算平均距离
    mean_method1 = elim_df["dist_method1"].mean()
    mean_method2 = elim_df["dist_method2"].mean()

    print(f"\n平均距离最差位置 (越小 = 预测越准确):")
    print(f"  方法1 (排名法): {mean_method1:.3f}")
    print(f"  方法2 (占比法): {mean_method2:.3f}")
    print(f"  差异: {abs(mean_method1 - mean_method2):.3f}")

    if abs(mean_method1 - mean_method2) < 0.01:
        conclusion = "两种方法预测效果几乎相同"
    elif mean_method1 < mean_method2:
        conclusion = "方法1 (排名法) 能更准确预测淘汰"
    else:
        conclusion = "方法2 (占比法) 能更准确预测淘汰"

    print(f"\n→ 结论: {conclusion}")

    # 按数据源分组统计
    print("\n" + "-" * 60)
    print("按数据源分组统计:")
    print("-" * 60)

    for source in elim_df["data_source"].dropna().unique():
        source_df = elim_df[elim_df["data_source"] == source]
        if len(source_df) > 0:
            m1 = source_df["dist_method1"].mean()
            m2 = source_df["dist_method2"].mean()
            better = "排名法" if m1 < m2 else "占比法" if m2 < m1 else "相同"
            print(
                f"  {source}: {len(source_df)} 个淘汰, 方法1={m1:.3f}, 方法2={m2:.3f} → {better}更优"
            )

    print(f"\n✓ 结果已保存:")
    print(f"  完整数据: {output_path / 'wilcoxon_full_data.csv'}")
    print(f"  淘汰者数据: {output_path / 'wilcoxon_eliminated.csv'}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
