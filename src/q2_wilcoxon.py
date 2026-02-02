"""
Wilcoxon 检验：比较排名法 vs 百分比法的预测效果

直接使用 q4_featured_data.csv 数据集，该数据集已包含：
- week{n}_combined_rank: 方法1（排名法）的综合排名
- week{n}_combined_pct: 方法2（百分比法）的综合百分比
- 缺失值用 -1 标记
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from typing import Tuple, Dict

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
    只保留有效数据（排除 -1 的缺失值）
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
            # 检查必要的列
            score_col = f"week{week}_score_sum"
            rank_col = f"week{week}_combined_rank"
            pct_col = f"week{week}_combined_pct"

            # 只处理有数据的周（score > 0 且 rank/pct 不为 -1）
            if score_col not in df.columns:
                continue

            score = row[score_col]
            combined_rank = row.get(rank_col, -1)
            combined_pct = row.get(pct_col, -1)

            # 跳过无效数据
            if pd.isna(score) or score <= 0:
                continue
            if combined_rank == -1 or combined_pct == -1:
                continue
            if pd.isna(combined_rank) or pd.isna(combined_pct):
                continue

            records.append(
                {
                    "celebrity_name": row["celebrity_name"],
                    "season": row["season"],
                    "week": week,
                    "judge_score": score,
                    "placement": placement,
                    "weeks_participated": weeks_participated,
                    "method1_combined_rank": combined_rank,  # 排名法：值越大越差
                    "method2_combined_pct": combined_pct,  # 百分比法：值越小越差
                }
            )

    long_df = pd.DataFrame(records)
    print(f"  ✓ 转换为长格式: {len(long_df)} 行 (已过滤 -1 缺失值)")
    return long_df


def calculate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每周的位置排名

    方法1 (排名法): combined_rank = judge_rank + audience_rank，值越小越好
    方法2 (百分比法): combined_pct = judge_pct + audience_pct，值越大越好

    需要将两者都转换为位置排名（pos），其中 pos 越大表示越危险
    """
    df = df.copy()

    # 方法1: combined_rank 越小越好，转换为位置排名
    # rank 小的人 pos 小（安全），rank 大的人 pos 大（危险）
    df["method1_pos"] = df.groupby(["season", "week"])["method1_combined_rank"].rank(
        ascending=True, method="average"
    )

    # 方法2: combined_pct 越大越好，转换为位置排名
    # pct 大的人 pos 小（安全），pct 小的人 pos 大（危险）
    df["method2_pos"] = df.groupby(["season", "week"])["method2_combined_pct"].rank(
        ascending=False, method="average"
    )

    # 每周参赛人数
    df["n_contestants"] = df.groupby(["season", "week"])["celebrity_name"].transform(
        "count"
    )

    return df


def identify_eliminated(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别淘汰者

    淘汰周 = weeks_participated（最后参与的周）
    前3名（placement <= 3）不算淘汰
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

    # 每周只取一个淘汰者（如果有多个，取第一个）
    elim_df = (
        elim_df.sort_values(["season", "week", "celebrity_name"])
        .groupby(["season", "week"], as_index=False)
        .head(1)
    )

    # 计算距离最差位置的距离（0 = 最差位置，值越小 = 预测越准确）
    elim_df["dist_method1"] = elim_df["n_contestants"] - elim_df["method1_pos"]
    elim_df["dist_method2"] = elim_df["n_contestants"] - elim_df["method2_pos"]

    return elim_df


def perform_wilcoxon_test_by_season(
    elim_df: pd.DataFrame,
) -> Tuple[float, float, int, pd.DataFrame]:
    """
    以赛季为单位执行 Wilcoxon 检验

    逻辑：
    1. 每个赛季计算所有淘汰者的距离中位数
    2. 得到 N 个赛季的配对数据 (method1_median, method2_median)
    3. 对赛季级别的数据执行 Wilcoxon 检验
    """
    # 按赛季分组，计算每个赛季的统计量
    season_stats = (
        elim_df.groupby("season")
        .agg(
            {
                "dist_method1": ["median", "mean", "count"],
                "dist_method2": ["median", "mean", "count"],
            }
        )
        .reset_index()
    )

    # 展平列名
    season_stats.columns = ["_".join(col).strip("_") for col in season_stats.columns]
    season_stats.rename(
        columns={
            "dist_method1_median": "method1_median",
            "dist_method1_mean": "method1_mean",
            "dist_method1_count": "n_eliminations",
            "dist_method2_median": "method2_median",
            "dist_method2_mean": "method2_mean",
        },
        inplace=True,
    )

    # 删除重复的 count 列
    if "dist_method2_count" in season_stats.columns:
        season_stats.drop(columns=["dist_method2_count"], inplace=True)

    # !!! 关键修改：直接对每个淘汰事件进行检验，而不是赛季中位数 !!!
    # 提取所有淘汰事件的配对距离
    x = elim_df["dist_method1"].to_numpy()
    y = elim_df["dist_method2"].to_numpy()

    # 检查有效性
    ok = np.isfinite(x) & np.isfinite(y)
    x_clean = x[ok]
    y_clean = y[ok]

    if len(x_clean) < 10:
        raise ValueError(
            f"没有足够的有效淘汰事件进行 Wilcoxon 检验 (需要至少10个，当前: {len(x_clean)})"
        )

    # 执行配对 Wilcoxon 符号秩检验
    # x_clean 和 y_clean 是配对的距离数据
    stat, p = wilcoxon(
        x_clean, y_clean, alternative="two-sided", zero_method="wilcox", method="auto"
    )

    return stat, p, len(x_clean), season_stats


def bootstrap_wilcoxon_test(
    elim_df: pd.DataFrame, n_iterations: int = 1000, sample_ratio: float = 0.8
) -> Dict:
    """
    使用 Bootstrap 方法验证 Wilcoxon 检验的稳定性

    参数:
        elim_df: 淘汰者数据
        n_iterations: Bootstrap 迭代次数
        sample_ratio: 每次抽样的比例
    """
    n_total = len(elim_df)
    n_sample = int(n_total * sample_ratio)

    print(f"\n开始 Bootstrap 验证:")
    print(f"  总样本数: {n_total}")
    print(f"  每次抽样: {n_sample} 个 ({sample_ratio*100:.0f}%)")
    print(f"  迭代次数: {n_iterations}")

    bootstrap_results = {
        "p_values": [],
        "statistics": [],
        "n_seasons": [],
        "method1_better": 0,
        "method2_better": 0,
        "no_difference": 0,
        "significant_count": 0,
        "mean_median_method1": [],
        "mean_median_method2": [],
    }

    for i in range(n_iterations):
        # 随机抽样（带放回）- 事件级别
        sample_indices = np.random.choice(n_total, size=n_sample, replace=True)
        sample_df = elim_df.iloc[sample_indices].copy()

        try:
            # 直接使用事件级别的距离数据（不按赛季聚合）
            x = sample_df["dist_method1"].to_numpy()
            y = sample_df["dist_method2"].to_numpy()

            ok = np.isfinite(x) & np.isfinite(y)
            x_clean = x[ok]
            y_clean = y[ok]

            if len(x_clean) < 3:
                continue

            # 配对样本检验（事件级别）
            stat, p = wilcoxon(
                x_clean,
                y_clean,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            bootstrap_results["p_values"].append(p)
            bootstrap_results["statistics"].append(stat)
            bootstrap_results["n_seasons"].append(len(x_clean))  # 改名：实际是事件数

            mean_m1 = np.mean(x_clean)
            mean_m2 = np.mean(y_clean)
            bootstrap_results["mean_median_method1"].append(mean_m1)
            bootstrap_results["mean_median_method2"].append(mean_m2)

            if abs(mean_m1 - mean_m2) < 0.01:
                bootstrap_results["no_difference"] += 1
            elif mean_m1 < mean_m2:
                bootstrap_results["method1_better"] += 1
            else:
                bootstrap_results["method2_better"] += 1

            if p < 0.05:
                bootstrap_results["significant_count"] += 1

        except Exception:
            continue

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{n_iterations}", end="\r")

    print(f"  进度: {n_iterations}/{n_iterations} - 完成!")

    successful_iterations = len(bootstrap_results["p_values"])
    bootstrap_results["successful_iterations"] = successful_iterations

    if successful_iterations > 0:
        bootstrap_results["p_values"] = np.array(bootstrap_results["p_values"])
        bootstrap_results["statistics"] = np.array(bootstrap_results["statistics"])
        bootstrap_results["mean_median_method1"] = np.array(
            bootstrap_results["mean_median_method1"]
        )
        bootstrap_results["mean_median_method2"] = np.array(
            bootstrap_results["mean_median_method2"]
        )

    return bootstrap_results


def main():
    """执行 Wilcoxon 检验比较两种评估方法"""
    print("=" * 70)
    print("Wilcoxon 检验：比较排名法 vs 百分比法")
    print("=" * 70)
    print("\n数据来源: q4_featured_data.csv")
    print("  - week{n}_combined_rank: 方法1（排名法综合排名）")
    print("  - week{n}_combined_pct: 方法2（百分比法综合百分比）")
    print("  - 缺失值用 -1 标记，分析时自动过滤")
    print("\n比较逻辑:")
    print("  方法1 (排名法): combined_rank 越大 → 越可能被淘汰")
    print("  方法2 (百分比法): combined_pct 越小 → 越可能被淘汰")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    df = load_data()

    # 2. 转换为长格式
    print("\n[2/5] 转换为长格式...")
    long_df = transform_to_long_format(df)
    print(
        f"  ✓ {long_df['celebrity_name'].nunique()} 个选手, {long_df['season'].nunique()} 个赛季"
    )

    # 3. 计算位置排名
    print("\n[3/5] 计算位置排名...")
    long_df = calculate_positions(long_df)
    print(f"  ✓ 位置排名计算完成")

    # 4. 识别淘汰者
    print("\n[4/5] 识别淘汰者...")
    long_df = identify_eliminated(long_df)
    elim_df = extract_elimination_data(long_df)
    print(f"  ✓ 识别 {len(elim_df)} 个淘汰事件")

    # 5. 执行 Wilcoxon 检验
    print("\n[5/5] 执行 Wilcoxon 配对符号秩检验...")
    stat, p_value, n_seasons, season_stats = perform_wilcoxon_test_by_season(elim_df)

    # 保存结果
    output_path = OUTPUT_DIR / "question2_res" / "wilcoxon"
    output_path.mkdir(parents=True, exist_ok=True)

    long_df.to_csv(
        output_path / "wilcoxon_full_data.csv", index=False, encoding="utf-8"
    )
    elim_df.to_csv(
        output_path / "wilcoxon_eliminated.csv", index=False, encoding="utf-8"
    )
    season_stats.to_csv(
        output_path / "wilcoxon_season_stats.csv", index=False, encoding="utf-8"
    )

    # 输出结果
    print("\n" + "=" * 70)
    print("Wilcoxon 配对符号秩检验结果（事件级别）")
    print("=" * 70)

    print(
        f"\n检验样本数: {n_seasons} 个淘汰事件 (每个事件有配对的 method1 和 method2 距离)"
    )
    print(f"检验统计量: {stat:.4f}")
    print(f"P 值: {p_value:.6f}")
    print(
        f"显著性: {'显著差异 (p < 0.05)' if p_value < 0.05 else '无显著差异 (p >= 0.05)'}"
    )

    if p_value < 0.05:
        print("\n⚠️  注意：p < 0.05 表示两种方法存在显著差异！")
    else:
        print("\n⚠️  注意：p >= 0.05 表示无法拒绝「两种方法相同」的假设")

    # 事件级别的描述性统计
    mean_method1 = elim_df["dist_method1"].mean()
    mean_method2 = elim_df["dist_method2"].mean()
    median_method1 = elim_df["dist_method1"].median()
    median_method2 = elim_df["dist_method2"].median()

    print(f"\n描述性统计 (所有淘汰事件):")
    print(f"\n  平均距离 (越小 = 预测越准确):")
    print(f"    方法1 (排名法): {mean_method1:.3f}")
    print(f"    方法2 (百分比法): {mean_method2:.3f}")
    if mean_method1 > 0:
        improvement_pct = ((mean_method1 - mean_method2) / mean_method1) * 100
        print(f"    方法2改进: {improvement_pct:.1f}%")

    print(f"\n  中位数距离:")
    print(f"    方法1 (排名法): {median_method1:.3f}")
    print(f"    方法2 (百分比法): {median_method2:.3f}")

    # 完美预测比例
    perfect_method1 = (elim_df["dist_method1"] == 0).sum()
    perfect_method2 = (elim_df["dist_method2"] == 0).sum()
    total_events = len(elim_df)

    print(f"\n  完美预测 (距离=0):")
    print(
        f"    方法1: {perfect_method1}/{total_events} ({perfect_method1/total_events*100:.1f}%)"
    )
    print(
        f"    方法2: {perfect_method2}/{total_events} ({perfect_method2/total_events*100:.1f}%)"
    )

    if mean_method1 < mean_method2:
        conclusion = "方法1 (排名法) 能更准确预测淘汰"
    elif mean_method2 < mean_method1:
        conclusion = "方法2 (百分比法) 能更准确预测淘汰"
    else:
        conclusion = "两种方法预测效果几乎相同"

    print(f"\n→ 描述性统计结论: {conclusion}")
    print(
        f"→ 统计检验结论: {'两种方法存在显著差异' if p_value < 0.05 else '无显著差异'}"
    )

    # 显示部分淘汰事件详情
    print("\n" + "-" * 70)
    print("淘汰事件样本 (前15个):")
    print("-" * 70)
    display_cols = [
        "season",
        "week",
        "celebrity_name",
        "method1_pos",
        "method2_pos",
        "dist_method1",
        "dist_method2",
    ]
    print(elim_df[display_cols].head(15).to_string(index=False))

    # ===== Bootstrap 验证 =====
    print("\n" + "=" * 70)
    print("Bootstrap 验证")
    print("=" * 70)

    bootstrap_results = bootstrap_wilcoxon_test(
        elim_df, n_iterations=1000, sample_ratio=0.8
    )

    successful = bootstrap_results["successful_iterations"]
    if successful > 0:
        print("\n" + "-" * 70)
        print("Bootstrap 结果汇总:")
        print("-" * 70)

        print(f"\n成功迭代次数: {successful}/{1000}")

        p_values = bootstrap_results["p_values"]
        print(f"\nP 值分布:")
        print(f"  均值: {np.mean(p_values):.6f}")
        print(f"  中位数: {np.median(p_values):.6f}")
        print(f"  标准差: {np.std(p_values):.6f}")
        print(
            f"  95% 置信区间: [{np.percentile(p_values, 2.5):.6f}, {np.percentile(p_values, 97.5):.6f}]"
        )

        sig_rate = bootstrap_results["significant_count"] / successful * 100
        print(f"\n显著性检验 (p < 0.05):")
        print(f"  显著次数: {bootstrap_results['significant_count']}/{successful}")
        print(f"  显著率: {sig_rate:.1f}%")

        total_compared = (
            bootstrap_results["method1_better"]
            + bootstrap_results["method2_better"]
            + bootstrap_results["no_difference"]
        )
        m1_rate = bootstrap_results["method1_better"] / total_compared * 100
        m2_rate = bootstrap_results["method2_better"] / total_compared * 100
        no_diff_rate = bootstrap_results["no_difference"] / total_compared * 100

        print(f"\n方法优劣统计:")
        print(
            f"  方法1 (排名法) 更优: {bootstrap_results['method1_better']} 次 ({m1_rate:.1f}%)"
        )
        print(
            f"  方法2 (百分比法) 更优: {bootstrap_results['method2_better']} 次 ({m2_rate:.1f}%)"
        )
        print(
            f"  无明显差异: {bootstrap_results['no_difference']} 次 ({no_diff_rate:.1f}%)"
        )

        mean_m1 = bootstrap_results["mean_median_method1"]
        mean_m2 = bootstrap_results["mean_median_method2"]

        print(f"\n事件距离的平均值分布 (Bootstrap 抽样):")
        print(f"  方法1 (排名法):")
        print(f"    均值: {np.mean(mean_m1):.3f}")
        print(
            f"    95% CI: [{np.percentile(mean_m1, 2.5):.3f}, {np.percentile(mean_m1, 97.5):.3f}]"
        )
        print(f"  方法2 (百分比法):")
        print(f"    均值: {np.mean(mean_m2):.3f}")
        print(
            f"    95% CI: [{np.percentile(mean_m2, 2.5):.3f}, {np.percentile(mean_m2, 97.5):.3f}]"
        )

        diff = mean_m1 - mean_m2
        print(f"\n方法差异 (方法1 - 方法2):")
        print(f"  均值: {np.mean(diff):.3f}")
        print(f"  中位数: {np.median(diff):.3f}")
        print(
            f"  95% CI: [{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]"
        )
        print(
            f"  方法1误差 > 方法2误差: {np.sum(diff > 0)}/{successful} 次 ({np.sum(diff > 0)/successful*100:.1f}%)"
        )

        # 保存 Bootstrap 结果
        bootstrap_summary = pd.DataFrame(
            {
                "metric": [
                    "successful_iterations",
                    "p_value_mean",
                    "p_value_median",
                    "p_value_std",
                    "significant_rate",
                    "method1_better",
                    "method2_better",
                    "no_difference",
                    "method1_event_mean_avg",
                    "method2_event_mean_avg",
                    "diff_mean",
                    "diff_median",
                    "method1_worse_rate",
                ],
                "value": [
                    successful,
                    np.mean(p_values),
                    np.median(p_values),
                    np.std(p_values),
                    sig_rate,
                    bootstrap_results["method1_better"],
                    bootstrap_results["method2_better"],
                    bootstrap_results["no_difference"],
                    np.mean(mean_m1),
                    np.mean(mean_m2),
                    np.mean(diff),
                    np.median(diff),
                    np.sum(diff > 0) / successful * 100,
                ],
            }
        )
        bootstrap_summary.to_csv(
            output_path / "wilcoxon_bootstrap_summary.csv",
            index=False,
            encoding="utf-8",
        )

        # 确定最终结论
        if m2_rate > 60:
            final_conclusion = "方法2 (百分比法) 显著优于方法1 (排名法)"
        elif m1_rate > 60:
            final_conclusion = "方法1 (排名法) 显著优于方法2 (百分比法)"
        else:
            final_conclusion = "两种方法预测效果接近"

        print("\n" + "=" * 70)
        print("综合结论:")
        print("=" * 70)
        print(f"\n基于 Bootstrap 验证 ({successful}次抽样):")
        print(f"  - 方法2优势率: {m2_rate:.1f}%")
        print(f"  - 方法1误差大于方法2的比例: {np.sum(diff > 0)/successful*100:.1f}%")
        print(f"\n→ 最终结论: {final_conclusion}")

    print(f"\n✓ 结果已保存:")
    print(f"  完整数据: {output_path / 'wilcoxon_full_data.csv'}")
    print(f"  淘汰者数据: {output_path / 'wilcoxon_eliminated.csv'}")
    print(f"  赛季统计: {output_path / 'wilcoxon_season_stats.csv'}")
    print(f"  Bootstrap 汇总: {output_path / 'wilcoxon_bootstrap_summary.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
