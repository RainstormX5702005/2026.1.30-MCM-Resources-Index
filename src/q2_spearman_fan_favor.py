import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Tuple, Dict

from configs.config import OUTPUT_DIR

# 配置输入文件路径
INPUT_FILE = OUTPUT_DIR / "question2_res" / "wilcoxon" / "wilcoxon_full_data.csv"


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    加载数据并进行基础校验
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 找不到文件 {file_path}。请确认文件路径正确。")

    # 必要的列检查
    required_columns = ["season", "week", "method1_pos", "method2_pos", "judge_score"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"数据错误: 缺少关键列 {missing}。请检查 CSV 文件头。")

    # 类型转换与清洗
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除含有空值的行
    initial_len = len(df)
    df = df.dropna(subset=required_columns).copy()
    print(f"Data loaded. Rows: {len(df)} (Dropped {initial_len - len(df)} NaN rows)")

    return df


def calculate_correlation_with_judges(
    df: pd.DataFrame,
    method_col: str,
    judge_col: str = "judge_score",
    min_contestants: int = 4,
) -> pd.DataFrame:
    """
    计算每种方法与评委分数的 Spearman 相关系数

    核心逻辑：
    - 如果 method 与 judge_score 高度正相关 → 该方法更依赖评委打分
    - 如果 method 与 judge_score 相关性较弱 → 该方法可能更反映其他因素（如粉丝投票）

    注意：这里用的是 judge_score（分数越高越好），所以：
    - 如果 method_pos（排名越小越好）与 judge_score 负相关，说明分数高的排名靠前 → 强依赖评委
    - 如果相关性弱，说明排名不完全由评委决定 → 可能更受粉丝影响

    Args:
        df: 数据框
        method_col: 待评估的方法排名列 (例如 'method1_pos' 或 'method2_pos')
        judge_col: 评委分数列
        min_contestants: 每周最少参赛人数

    Returns:
        包含每周 correlation (rho) 的 DataFrame
    """
    results = []

    for (season, week), group in df.groupby(["season", "week"]):
        valid_data = group[[method_col, judge_col]].dropna()
        n = len(valid_data)

        if n < min_contestants:
            continue

        # 计算 Spearman 相关系数
        # 注意：method_pos 越小越好，judge_score 越大越好
        # 所以正常情况下应该是负相关（分数高的排名靠前）
        rho, p_value = spearmanr(valid_data[method_col], valid_data[judge_col])

        results.append(
            {
                "season": season,
                "week": week,
                "n_contestants": n,
                "rho": rho,
                "p_value": p_value,
                "abs_rho": abs(rho),  # 相关性强度（不考虑方向）
            }
        )

    return pd.DataFrame(results)


def calculate_residual_correlation(
    df: pd.DataFrame,
    method_col: str,
    judge_col: str = "judge_score",
    placement_col: str = "placement",
    min_contestants: int = 4,
) -> pd.DataFrame:
    """
    计算"去除评委影响后"的相关性分析

    思路：
    1. 先看 judge_score 与 placement 的关系（评委打分对最终结果的影响）
    2. 计算残差：actual_placement - predicted_by_judges
    3. 看哪个方法能更好地解释这个残差

    如果一个方法能解释更多残差 → 说明它捕捉到了评委分数之外的因素（如粉丝投票）
    """
    results = []

    for (season, week), group in df.groupby(["season", "week"]):
        valid_data = group[[method_col, judge_col, placement_col]].dropna()
        n = len(valid_data)

        if n < min_contestants:
            continue

        # 评委分数与最终排名的相关性（基准）
        rho_judge_placement, _ = spearmanr(
            valid_data[judge_col], valid_data[placement_col]
        )

        # 方法预测与最终排名的相关性
        rho_method_placement, _ = spearmanr(
            valid_data[method_col], valid_data[placement_col]
        )

        # 方法预测与评委分数的相关性
        rho_method_judge, p_val = spearmanr(
            valid_data[method_col], valid_data[judge_col]
        )

        # 计算"超出评委影响"的解释能力
        # 简化指标：如果方法能预测最终结果，但不完全依赖评委分数
        # 则 abs(rho_method_placement) 高但 abs(rho_method_judge) 相对较低
        independence_score = abs(rho_method_placement) - abs(rho_method_judge)

        results.append(
            {
                "season": season,
                "week": week,
                "n_contestants": n,
                "rho_method_judge": rho_method_judge,
                "rho_method_placement": rho_method_placement,
                "rho_judge_placement": rho_judge_placement,
                "p_value": p_val,
                "independence_score": independence_score,  # 越高说明越不依赖评委
            }
        )

    return pd.DataFrame(results)


def compare_fan_favor(
    df_rank: pd.DataFrame, df_share: pd.DataFrame
) -> Tuple[Dict, pd.DataFrame]:
    """
    对比两种方法谁更 favor fan votes

    判断标准：
    1. 与评委分数的相关性：越低 → 越不依赖评委 → 可能更依赖粉丝
    2. Independence score：越高 → 越能解释"评委之外"的因素 → 可能是粉丝影响
    """
    merged = pd.merge(
        df_rank,
        df_share,
        on=["season", "week"],
        suffixes=("_rank", "_share"),
    )

    # 相关性差异（Share - Rank）
    # 如果为负，说明 Share 与评委相关性更低 → Share 更不依赖评委
    merged["judge_corr_diff"] = abs(merged["rho_method_judge_share"]) - abs(
        merged["rho_method_judge_rank"]
    )

    # Independence score 差异
    merged["independence_diff"] = (
        merged["independence_score_share"] - merged["independence_score_rank"]
    )

    stats = {
        "n_weeks": len(merged),
        # 与评委分数的平均相关性（绝对值）
        "mean_judge_corr_rank": abs(merged["rho_method_judge_rank"]).mean(),
        "mean_judge_corr_share": abs(merged["rho_method_judge_share"]).mean(),
        # Independence scores
        "mean_independence_rank": merged["independence_score_rank"].mean(),
        "mean_independence_share": merged["independence_score_share"].mean(),
        # 哪个方法更独立于评委？
        "share_less_judge_dependent": (merged["judge_corr_diff"] < 0).mean() * 100,  # %
        "rank_less_judge_dependent": (merged["judge_corr_diff"] > 0).mean() * 100,
        # 哪个方法 independence score 更高？
        "share_more_independent": (merged["independence_diff"] > 0).mean() * 100,
        "rank_more_independent": (merged["independence_diff"] < 0).mean() * 100,
        # 平均差异
        "mean_judge_corr_diff": merged["judge_corr_diff"].mean(),
        "mean_independence_diff": merged["independence_diff"].mean(),
    }

    return stats, merged


def print_report(stats: Dict, corr_rank: pd.DataFrame, corr_share: pd.DataFrame):
    """
    打印分析报告：哪个方法更 favor fan votes？
    """
    print("\n" + "=" * 60)
    print("🎭  FAN VOTES FAVORITISM ANALYSIS: SHARE vs RANK")
    print("=" * 60)
    print(f"Total Weeks Analyzed: {stats['n_weeks']}")
    print("-" * 60)

    print(f"\n1. Correlation with Judge Scores (Lower = Less Judge-Dependent):")
    print(f"   - Rank Model:  {stats['mean_judge_corr_rank']:.4f}")
    print(f"   - Share Model: {stats['mean_judge_corr_share']:.4f}")

    if stats["mean_judge_corr_rank"] > stats["mean_judge_corr_share"]:
        diff = stats["mean_judge_corr_rank"] - stats["mean_judge_corr_share"]
        print(f"   → Share Model is {diff:.4f} LESS dependent on judges ✓")
    else:
        diff = stats["mean_judge_corr_share"] - stats["mean_judge_corr_rank"]
        print(f"   → Rank Model is {diff:.4f} LESS dependent on judges ✓")

    print("-" * 60)
    print(f"\n2. Independence Score (Higher = More Non-Judge Factors):")
    print(f"   - Rank Model:  {stats['mean_independence_rank']:.4f}")
    print(f"   - Share Model: {stats['mean_independence_share']:.4f}")

    if stats["mean_independence_share"] > stats["mean_independence_rank"]:
        diff = stats["mean_independence_share"] - stats["mean_independence_rank"]
        print(f"   → Share Model explains {diff:.4f} MORE non-judge factors ✓")
    else:
        diff = stats["mean_independence_rank"] - stats["mean_independence_share"]
        print(f"   → Rank Model explains {diff:.4f} MORE non-judge factors ✓")

    print("-" * 60)
    print(f"\n3. Week-by-Week Comparison:")
    print(
        f"   - Share Model less judge-dependent: {stats['share_less_judge_dependent']:.1f}%"
    )
    print(
        f"   - Rank Model less judge-dependent:  {stats['rank_less_judge_dependent']:.1f}%"
    )
    print()
    print(f"   - Share Model more independent: {stats['share_more_independent']:.1f}%")
    print(f"   - Rank Model more independent:  {stats['rank_more_independent']:.1f}%")

    print("-" * 60)
    print(f"\n4. Statistical Significance:")
    sig_rank = (corr_rank["p_value"] < 0.05).mean() * 100
    sig_share = (corr_share["p_value"] < 0.05).mean() * 100
    print(
        f"   - Rank-Judge correlation significant (p<0.05):  {sig_rank:.1f}% of weeks"
    )
    print(
        f"   - Share-Judge correlation significant (p<0.05): {sig_share:.1f}% of weeks"
    )

    print("=" * 60)

    # 综合结论
    print("\n📊 CONCLUSION:")
    print("-" * 60)

    # 判断哪个方法更 favor fan votes
    evidence_for_share = 0
    evidence_for_rank = 0

    if stats["mean_judge_corr_share"] < stats["mean_judge_corr_rank"]:
        evidence_for_share += 1
        print("✓ Share Model has WEAKER correlation with judge scores")
    else:
        evidence_for_rank += 1
        print("✓ Rank Model has WEAKER correlation with judge scores")

    if stats["mean_independence_share"] > stats["mean_independence_rank"]:
        evidence_for_share += 1
        print("✓ Share Model explains MORE non-judge factors")
    else:
        evidence_for_rank += 1
        print("✓ Rank Model explains MORE non-judge factors")

    if stats["share_less_judge_dependent"] > stats["rank_less_judge_dependent"]:
        evidence_for_share += 1
        print(
            f"✓ Share Model is less judge-dependent in {stats['share_less_judge_dependent']:.1f}% of weeks"
        )
    else:
        evidence_for_rank += 1
        print(
            f"✓ Rank Model is less judge-dependent in {stats['rank_less_judge_dependent']:.1f}% of weeks"
        )

    print("-" * 60)
    if evidence_for_share > evidence_for_rank:
        print(
            f"\n🎯 ANSWER: The SHARE Model seems to favor fan votes MORE than Rank Model"
        )
        print(
            "   It is less dependent on judge scores and captures more non-judge factors."
        )
    elif evidence_for_rank > evidence_for_share:
        print(
            f"\n🎯 ANSWER: The RANK Model seems to favor fan votes MORE than Share Model"
        )
        print(
            "   It is less dependent on judge scores and captures more non-judge factors."
        )
    else:
        print(f"\n🎯 ANSWER: Both models show SIMILAR dependence on fan votes")
        print("   The difference is not substantial enough to draw a clear conclusion.")

    print("=" * 60)


def main():
    """执行 Fan Votes Favoritism 分析"""
    # 1. 加载数据
    try:
        df = load_and_validate_data(INPUT_FILE)
    except Exception as e:
        print(e)
        return

    # 2. 计算与评委分数的相关性
    print("\n=== Analyzing Rank Model's dependence on judge scores ===")
    corr_rank = calculate_residual_correlation(df, method_col="method1_pos")

    print("\n=== Analyzing Share Model's dependence on judge scores ===")
    corr_share = calculate_residual_correlation(df, method_col="method2_pos")

    # 3. 对比分析
    print("\n=== Comparing which method favors fan votes more ===")
    stats, comparison_df = compare_fan_favor(corr_rank, corr_share)

    # 4. 输出报告
    print_report(stats, corr_rank, corr_share)

    # 5. 保存结果
    output_dir = OUTPUT_DIR / "question2_res" / "spearman"
    output_dir.mkdir(parents=True, exist_ok=True)

    corr_rank.to_csv(
        output_dir / "fan_favor_rank_method.csv", index=False, encoding="utf-8"
    )
    corr_share.to_csv(
        output_dir / "fan_favor_share_method.csv", index=False, encoding="utf-8"
    )
    comparison_df.to_csv(
        output_dir / "fan_favor_comparison.csv", index=False, encoding="utf-8"
    )

    print(f"\n✓ Detailed results saved to: {output_dir}")
    print(f"  - fan_favor_rank_method.csv")
    print(f"  - fan_favor_share_method.csv")
    print(f"  - fan_favor_comparison.csv")


if __name__ == "__main__":
    main()
