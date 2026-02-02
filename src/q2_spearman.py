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
    required_columns = ["season", "week", "method1_pos", "method2_pos", "placement"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"数据错误: 缺少关键列 {missing}。请检查 CSV 文件头。")

    # 类型转换与清洗
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除含有空值的行 (保证对比公平)
    initial_len = len(df)
    df = df.dropna(subset=required_columns).copy()
    print(f"Data loaded. Rows: {len(df)} (Dropped {initial_len - len(df)} NaN rows)")

    return df


def prepare_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    准备真实排名（Ground Truth）

    使用最终名次 placement 作为真实排名：
    - placement 越小越好（1st 最好）
    - 在每周内，根据 placement 建立真实的优劣顺序
    """
    df = df.copy()

    # 在每周内，根据 placement 计算真实排名
    # placement 小的应该排名靠前（值小）
    df["true_rank"] = df.groupby(["season", "week"])["placement"].rank(
        ascending=True, method="average"
    )

    return df


def calculate_weekly_correlation(
    df: pd.DataFrame,
    method_col: str,
    truth_col: str = "true_rank",
    min_contestants: int = 4,
) -> pd.DataFrame:
    """
    计算每一周的 Spearman 相关系数

    Args:
        df: 数据框
        method_col: 待评估的方法排名列 (例如 'method1_pos' 或 'method2_pos')
        truth_col: 真实排名列 (默认为 'true_rank')
        min_contestants: 每周最少参赛人数 (少于此数不计算)

    Returns:
        包含每周 correlation (rho) 的 DataFrame
    """
    results = []

    # 按赛季和周分组遍历
    for (season, week), group in df.groupby(["season", "week"]):
        # 再次确保无空值
        valid_data = group[[method_col, truth_col]].dropna()
        n = len(valid_data)

        if n < min_contestants:
            continue

        # 计算 Spearman 相关系数
        rho, p_value = spearmanr(valid_data[method_col], valid_data[truth_col])

        results.append(
            {
                "season": season,
                "week": week,
                "n_contestants": n,
                "rho": rho,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(results)


def compare_methods(
    df_corr_a: pd.DataFrame, df_corr_b: pd.DataFrame
) -> Tuple[Dict, pd.DataFrame]:
    """
    对比两种方法的相关系数结果
    """
    # 合并两组结果，确保只比较同一周的数据 (对齐)
    merged = pd.merge(
        df_corr_a, df_corr_b, on=["season", "week"], suffixes=("_rank", "_share")
    )

    # 计算差异 (Share模型 - Rank模型)
    # rho 越高越好 (越接近 1 说明越准确)
    merged["diff"] = merged["rho_share"] - merged["rho_rank"]

    stats = {
        "n_weeks_compared": len(merged),
        "mean_rho_rank": merged["rho_rank"].mean(),
        "mean_rho_share": merged["rho_share"].mean(),
        "mean_diff": merged["diff"].mean(),
        "median_diff": merged["diff"].median(),
        "win_rate_rank": (merged["diff"] < 0).mean(),  # Rank 模型胜出的比例
        "win_rate_share": (merged["diff"] > 0).mean(),  # Share 模型胜出的比例
        # p 值统计
        "mean_p_rank": merged["p_value_rank"].mean(),
        "mean_p_share": merged["p_value_share"].mean(),
        "significant_rank": (merged["p_value_rank"] < 0.05).mean(),  # 显著相关比例
        "significant_share": (merged["p_value_share"] < 0.05).mean(),  # 显著相关比例
        "very_significant_rank": (merged["p_value_rank"] < 0.01).mean(),  # 高度显著比例
        "very_significant_share": (
            merged["p_value_share"] < 0.01
        ).mean(),  # 高度显著比例
    }

    return stats, merged


def print_report(stats: Dict):
    """
    打印漂亮的分析报告
    """
    print("\n" + "=" * 50)
    print("🏆  MODEL COMPARISON REPORT: SHARE vs RANK")
    print("=" * 50)
    print(f"Total Weeks Analyzed: {stats['n_weeks_compared']}")
    print("-" * 30)
    print(f"1. Average Correlation (Higher is Better):")
    print(f"   - Share Model (Ours): {stats['mean_rho_share']:.4f}")
    print(f"   - Rank Model (Official): {stats['mean_rho_rank']:.4f}")
    print("-" * 30)
    print(f"2. Statistical Significance (p-values):")
    print(f"   - Share Model Mean p-value: {stats['mean_p_share']:.6f}")
    print(f"   - Rank Model Mean p-value:  {stats['mean_p_rank']:.6f}")
    print(
        f"   - Share Model Significant (p<0.05): {stats['significant_share']*100:.1f}%"
    )
    print(
        f"   - Rank Model Significant (p<0.05):  {stats['significant_rank']*100:.1f}%"
    )
    print(
        f"   - Share Model Very Significant (p<0.01): {stats['very_significant_share']*100:.1f}%"
    )
    print(
        f"   - Rank Model Very Significant (p<0.01):  {stats['very_significant_rank']*100:.1f}%"
    )
    print("-" * 30)
    print(f"3. Direct Head-to-Head Comparison:")
    print(f"   - Mean Improvement: {stats['mean_diff']:.4f}")
    print(f"   - Median Improvement: {stats['median_diff']:.4f}")
    print("-" * 30)
    print(f"4. Win Rate (Which model was more accurate per week?):")
    print(f"   - Share Model Wins: {stats['win_rate_share']*100:.1f}%")
    print(f"   - Rank Model Wins:  {stats['win_rate_rank']*100:.1f}%")
    print("=" * 50)

    if stats["mean_diff"] > 0:
        print("✅ CONCLUSION: The Share-based model is more accurate.")
        print("   It better reflects the true contestant rankings.")
    else:
        print("❌ CONCLUSION: The Rank-based model is more accurate.")


def main():
    """执行 Spearman 相关性分析"""
    # 1. 加载数据
    try:
        df = load_and_validate_data(INPUT_FILE)
    except Exception as e:
        print(e)
        return

    # 2. 准备真实排名
    print("\nPreparing ground truth rankings...")
    df = prepare_ground_truth(df)
    print(f"  Ground truth based on final placement")

    # 3. 计算每种方法的每周表现
    print("\nCalculating correlations for Rank Model (Method 1)...")
    corr_rank = calculate_weekly_correlation(
        df, method_col="method1_pos", truth_col="true_rank"
    )

    print("Calculating correlations for Share Model (Method 2)...")
    corr_share = calculate_weekly_correlation(
        df, method_col="method2_pos", truth_col="true_rank"
    )

    # 4. 对比两种方法
    print("Comparing methods...")
    stats, comparison_df = compare_methods(corr_rank, corr_share)

    # 5. 输出报告
    print_report(stats)

    # 6. 保存详细对比结果
    output_dir = OUTPUT_DIR / "question2_res" / "spearman"

    corr_rank.to_csv(
        output_dir / "spearman_rank_method.csv", index=False, encoding="utf-8"
    )
    corr_share.to_csv(
        output_dir / "spearman_share_method.csv", index=False, encoding="utf-8"
    )
    comparison_df.to_csv(
        output_dir / "spearman_comparison.csv", index=False, encoding="utf-8"
    )

    print(f"\n✓ Detailed results saved:")
    print(f"  - spearman_rank_method.csv")
    print(f"  - spearman_share_method.csv")
    print(f"  - spearman_comparison.csv")


if __name__ == "__main__":
    main()
