"""
对争议选手进行局部 Wilcoxon 检验（v2版本）

使用 q4_featured_data.csv 数据，基于 q2_wilcoxon_v2 的逻辑
找到附件中提到的争议选手，并补充到更多，
然后对这些选手使用事件级别的 Wilcoxon 检验
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from typing import List, Tuple, Dict

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


def get_controversial_contestants(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """
    返回争议选手列表（姓名，赛季）

    基于附件中提到的争议选手：
    - Jerry Rice (season 2) - 亚军但评委打分最低持续5周
    - Billy Ray Cyrus (season 4) - 第5名但评委打分倒数第一持续6周
    - Bristol Palin (season 11) - 第3名但评委打分最低12次
    - Bobby Bones (season 27) - 冠军但评委打分持续偏低

    然后自动补充更多的争议选手
    """
    # 附件中明确提到的争议选手
    controversial = [
        ("Jerry Rice", 2),
        ("Billy Ray Cyrus", 4),
        ("Bristol Palin", 11),
        ("Bobby Bones", 27),
    ]

    # 从数据中找到更多的争议选手
    # 定义争议标准：相对排名较好但评委分数较低的选手
    # 使用 relative_placement (0-1, 越小越好) 和平均评委分数

    # 计算每个选手的平均评委分数
    score_cols = [f"week{i}_score_sum" for i in range(1, 12)]
    df_copy = df.copy()

    # 对每个选手计算平均评委分数
    for col in score_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(-1, np.nan)

    df_copy["avg_judge_score"] = df_copy[score_cols].mean(axis=1, skipna=True)

    # 筛选条件：
    # 1. relative_placement < 0.3 (前30%，成绩较好)
    # 2. avg_judge_score 在其赛季中处于后50%
    controversial_candidates = []

    for season in df_copy["season"].unique():
        season_df = df_copy[df_copy["season"] == season].copy()
        season_median_score = season_df["avg_judge_score"].median()

        # 找到成绩好但评委分数低的选手
        mask = (season_df["relative_placement"] < 0.3) & (  # 最终排名前30%
            season_df["avg_judge_score"] < season_median_score
        )  # 评委分数低于中位数

        candidates = season_df[mask][
            ["celebrity_name", "season", "relative_placement", "avg_judge_score"]
        ]
        for _, row in candidates.iterrows():
            name = row["celebrity_name"]
            s = int(row["season"])
            # 避免重复
            if (name, s) not in controversial:
                controversial_candidates.append(
                    (name, s, row["relative_placement"], row["avg_judge_score"])
                )

    # 按 relative_placement 排序（越小越好），选择前10个
    controversial_candidates.sort(key=lambda x: x[2])
    for name, season, _, _ in controversial_candidates[:10]:
        controversial.append((name, season))

    print(f"\n选定的争议选手共 {len(controversial)} 个：")
    for i, (name, season) in enumerate(controversial, 1):
        print(f"  {i}. {name} (Season {season})")

    return controversial


def transform_to_long_format_for_contestants(
    df: pd.DataFrame, contestant_list: List[Tuple[str, int]]
) -> pd.DataFrame:
    """
    将指定选手的宽格式数据转换为长格式
    """
    records = []

    # 创建选手-赛季的查找集合
    contestant_set = set(contestant_list)

    for _, row in df.iterrows():
        # 只处理指定的选手
        if (row["celebrity_name"], row["season"]) not in contestant_set:
            continue

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
            rank_col = f"week{week}_combined_rank"
            pct_col = f"week{week}_combined_pct"

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
                    "method1_combined_rank": combined_rank,
                    "method2_combined_pct": combined_pct,
                }
            )

    long_df = pd.DataFrame(records)
    print(f"  ✓ 转换为长格式: {len(long_df)} 行")
    return long_df


def calculate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每周的位置排名
    """
    df = df.copy()

    # 方法1 (排名法): combined_rank 越小越好 → 排名时 ascending=True
    df["method1_pos"] = df.groupby(["season", "week"])["method1_combined_rank"].rank(
        ascending=True, method="average"
    )

    # 方法2 (百分比法): combined_pct 越大越好 → 排名时 ascending=False
    df["method2_pos"] = df.groupby(["season", "week"])["method2_combined_pct"].rank(
        ascending=False, method="average"
    )

    return df


def identify_eliminated(df: pd.DataFrame) -> pd.DataFrame:
    """
    标记淘汰者
    """
    df = df.copy()
    df["is_eliminated"] = (df["week"] == df["weeks_participated"]).astype(int)
    return df


def extract_elimination_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取淘汰事件数据
    """
    elim_df = df[df["is_eliminated"] == 1].copy()

    # 计算每周参赛人数
    n_contestants = (
        df.groupby(["season", "week"])["celebrity_name"].count().rename("n_contestants")
    )
    elim_df = elim_df.merge(
        n_contestants, left_on=["season", "week"], right_index=True, how="left"
    )

    # 计算距离最差位置的距离（0 = 完美预测）
    elim_df["dist_method1"] = elim_df["n_contestants"] - elim_df["method1_pos"]
    elim_df["dist_method2"] = elim_df["n_contestants"] - elim_df["method2_pos"]

    return elim_df


def perform_wilcoxon_test(elim_df: pd.DataFrame) -> Tuple[float, float, int]:
    """
    执行事件级别的 Wilcoxon 配对检验
    """
    x = elim_df["dist_method1"].to_numpy()
    y = elim_df["dist_method2"].to_numpy()

    ok = np.isfinite(x) & np.isfinite(y)
    x_clean = x[ok]
    y_clean = y[ok]

    if len(x_clean) < 3:
        return np.nan, np.nan, len(x_clean)

    stat, p = wilcoxon(
        x_clean, y_clean, alternative="two-sided", zero_method="wilcox", method="auto"
    )

    return stat, p, len(x_clean)


def bootstrap_wilcoxon_test(
    elim_df: pd.DataFrame, n_iterations: int = 1000, sample_ratio: float = 0.8
) -> Dict:
    """
    Bootstrap 验证（事件级别）
    """
    n_total = len(elim_df)
    n_sample = int(n_total * sample_ratio)

    print(f"\n  Bootstrap 验证:")
    print(f"    总样本数: {n_total}")
    print(f"    每次抽样: {n_sample} 个 ({sample_ratio*100:.0f}%)")
    print(f"    迭代次数: {n_iterations}")

    bootstrap_results = {
        "p_values": [],
        "statistics": [],
        "method1_better": 0,
        "method2_better": 0,
        "no_difference": 0,
        "significant_count": 0,
        "mean_method1": [],
        "mean_method2": [],
    }

    for i in range(n_iterations):
        sample_indices = np.random.choice(n_total, size=n_sample, replace=True)
        sample_df = elim_df.iloc[sample_indices].copy()

        try:
            x = sample_df["dist_method1"].to_numpy()
            y = sample_df["dist_method2"].to_numpy()

            ok = np.isfinite(x) & np.isfinite(y)
            x_clean = x[ok]
            y_clean = y[ok]

            if len(x_clean) < 3:
                continue

            stat, p = wilcoxon(
                x_clean,
                y_clean,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            bootstrap_results["p_values"].append(p)
            bootstrap_results["statistics"].append(stat)

            mean_m1 = np.mean(x_clean)
            mean_m2 = np.mean(y_clean)
            bootstrap_results["mean_method1"].append(mean_m1)
            bootstrap_results["mean_method2"].append(mean_m2)

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
            print(f"    进度: {i+1}/{n_iterations}", end="\r")

    print(f"    进度: {n_iterations}/{n_iterations} - 完成!")

    return bootstrap_results


def analyze_by_contestant(
    elim_df: pd.DataFrame,
    long_df: pd.DataFrame,
    controversial_list: List[Tuple[str, int]],
):
    """
    按每个争议选手分析其表现
    """
    print("\n" + "=" * 80)
    print("各争议选手的详细分析")
    print("=" * 80)

    results = []

    for name, season in controversial_list:
        # 选手的所有数据
        contestant_df = long_df[
            (long_df["celebrity_name"] == name) & (long_df["season"] == season)
        ].copy()

        # 选手的淘汰事件（如果被淘汰）
        contestant_elim = elim_df[
            (elim_df["celebrity_name"] == name) & (elim_df["season"] == season)
        ]

        if len(contestant_df) == 0:
            print(f"\n{name} (Season {season}): 无数据")
            continue

        weeks = contestant_df["week"].nunique()
        avg_judge_score = contestant_df["judge_score"].mean()
        placement = contestant_df["placement"].iloc[0]

        avg_method1_pos = contestant_df["method1_pos"].mean()
        avg_method2_pos = contestant_df["method2_pos"].mean()

        # 计算每周的平均参赛人数
        avg_n_contestants = (
            long_df[
                (long_df["season"] == season)
                & (long_df["week"].isin(contestant_df["week"]))
            ]
            .groupby("week")["celebrity_name"]
            .count()
            .mean()
        )

        print(f"\n{name} (Season {season}) - 最终排名第{placement}名:")
        print(f"  参赛周数: {weeks}")
        print(f"  平均评委分数: {avg_judge_score:.2f}")
        print(f"  平均每周参赛人数: {avg_n_contestants:.1f}")
        print(f"  方法1平均位置: {avg_method1_pos:.2f}")
        print(f"  方法2平均位置: {avg_method2_pos:.2f}")

        # 如果有淘汰数据
        if len(contestant_elim) > 0:
            dist1 = contestant_elim["dist_method1"].iloc[0]
            dist2 = contestant_elim["dist_method2"].iloc[0]
            print(f"  淘汰周预测距离: 方法1={dist1:.2f}, 方法2={dist2:.2f}")

        results.append(
            {
                "celebrity_name": name,
                "season": season,
                "placement": placement,
                "weeks": weeks,
                "avg_judge_score": avg_judge_score,
                "avg_method1_pos": avg_method1_pos,
                "avg_method2_pos": avg_method2_pos,
                "avg_n_contestants": avg_n_contestants,
            }
        )

    return pd.DataFrame(results)


def main():
    """执行争议选手的 Wilcoxon 检验"""
    print("=" * 80)
    print("争议选手的局部 Wilcoxon 检验 (v2)")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    df = load_data()

    # 2. 获取争议选手列表
    print("\n[2/6] 获取争议选手列表...")
    controversial_list = get_controversial_contestants(df)

    # 3. 转换为长格式（仅争议选手）
    print("\n[3/6] 转换为长格式（仅争议选手）...")
    long_df = transform_to_long_format_for_contestants(df, controversial_list)

    # 4. 计算位置排名
    print("\n[4/6] 计算位置排名...")
    long_df = calculate_positions(long_df)
    print(f"  ✓ 位置排名计算完成")

    # 5. 识别淘汰者
    print("\n[5/6] 识别淘汰者...")
    long_df = identify_eliminated(long_df)
    elim_df = extract_elimination_data(long_df)
    print(f"  ✓ 识别 {len(elim_df)} 个淘汰事件（争议选手）")

    # 6. 执行 Wilcoxon 检验
    print("\n[6/6] 执行 Wilcoxon 配对符号秩检验...")
    stat, p_value, n_events = perform_wilcoxon_test(elim_df)

    # 输出结果
    print("\n" + "=" * 80)
    print("Wilcoxon 配对符号秩检验结果（事件级别 - 争议选手）")
    print("=" * 80)

    print(f"\n检验样本数: {n_events} 个淘汰事件")
    print(f"检验统计量: {stat:.4f}")
    print(f"P 值: {p_value:.6f}")
    print(
        f"显著性: {'显著差异 (p < 0.05)' if p_value < 0.05 else '无显著差异 (p >= 0.05)'}"
    )

    # 描述性统计
    mean_method1 = elim_df["dist_method1"].mean()
    mean_method2 = elim_df["dist_method2"].mean()
    median_method1 = elim_df["dist_method1"].median()
    median_method2 = elim_df["dist_method2"].median()

    print(f"\n描述性统计 (争议选手淘汰事件):")
    print(f"\n  平均距离 (越小 = 预测越准确):")
    print(f"    方法1 (排名法): {mean_method1:.3f}")
    print(f"    方法2 (百分比法): {mean_method2:.3f}")
    if mean_method1 > 0:
        improvement_pct = ((mean_method1 - mean_method2) / mean_method1) * 100
        print(f"    方法2改进: {improvement_pct:.1f}%")

    print(f"\n  中位数距离:")
    print(f"    方法1 (排名法): {median_method1:.3f}")
    print(f"    方法2 (百分比法): {median_method2:.3f}")

    perfect_method1 = (elim_df["dist_method1"] == 0).sum()
    perfect_method2 = (elim_df["dist_method2"] == 0).sum()

    print(f"\n  完美预测 (距离=0):")
    print(
        f"    方法1: {perfect_method1}/{n_events} ({perfect_method1/n_events*100:.1f}%)"
    )
    print(
        f"    方法2: {perfect_method2}/{n_events} ({perfect_method2/n_events*100:.1f}%)"
    )

    # Bootstrap 验证
    if n_events >= 10:
        print("\n" + "=" * 80)
        print("Bootstrap 验证")
        print("=" * 80)

        bootstrap_results = bootstrap_wilcoxon_test(elim_df, n_iterations=1000)

        successful = len(bootstrap_results["p_values"])
        if successful > 0:
            p_values = np.array(bootstrap_results["p_values"])
            sig_rate = bootstrap_results["significant_count"] / successful * 100

            print(f"\n  成功迭代次数: {successful}/1000")
            print(f"\n  P 值分布:")
            print(f"    均值: {np.mean(p_values):.6f}")
            print(f"    中位数: {np.median(p_values):.6f}")
            print(f"    标准差: {np.std(p_values):.6f}")
            print(
                f"    95% CI: [{np.percentile(p_values, 2.5):.6f}, {np.percentile(p_values, 97.5):.6f}]"
            )
            print(f"\n  显著性检验 (p < 0.05):")
            print(
                f"    显著次数: {bootstrap_results['significant_count']}/{successful}"
            )
            print(f"    显著率: {sig_rate:.1f}%")

            total_compared = (
                bootstrap_results["method1_better"]
                + bootstrap_results["method2_better"]
                + bootstrap_results["no_difference"]
            )
            if total_compared > 0:
                m2_rate = bootstrap_results["method2_better"] / total_compared * 100
                print(f"\n  方法优劣统计:")
                print(
                    f"    方法1 (排名法) 更优: {bootstrap_results['method1_better']} 次"
                )
                print(
                    f"    方法2 (百分比法) 更优: {bootstrap_results['method2_better']} 次 ({m2_rate:.1f}%)"
                )
                print(f"    无明显差异: {bootstrap_results['no_difference']} 次")
    else:
        print("\n⚠️  样本量太小，跳过 Bootstrap 验证")

    # 按选手分析
    contestant_stats = analyze_by_contestant(elim_df, long_df, controversial_list)

    # 保存结果
    output_path = OUTPUT_DIR / "question2_res" / "abnormal"
    output_path.mkdir(parents=True, exist_ok=True)

    long_df.to_csv(
        output_path / "abnormal_wilcoxon_full_data.csv",
        index=False,
        encoding="utf-8",
    )
    elim_df.to_csv(
        output_path / "abnormal_wilcoxon_eliminated.csv",
        index=False,
        encoding="utf-8",
    )
    contestant_stats.to_csv(
        output_path / "abnormal_wilcoxon_contestant_stats.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("综合结论（争议选手）:")
    print("=" * 80)

    if p_value < 0.05:
        if mean_method1 < mean_method2:
            print("→ 对于争议选手，方法1 (排名法) 显著优于方法2")
        else:
            print("→ 对于争议选手，方法2 (百分比法) 显著优于方法1")
    else:
        print("→ 对于争议选手，两种方法无显著差异")

    print(f"\n✓ 结果已保存:")
    print(f"  完整数据: {output_path / 'abnormal_wilcoxon_v2_full_data.csv'}")
    print(f"  淘汰者数据: {output_path / 'abnormal_wilcoxon_v2_eliminated.csv'}")
    print(f"  选手统计: {output_path / 'abnormal_wilcoxon_v2_contestant_stats.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
