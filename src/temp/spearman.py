import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Tuple, Dict

# 配置输入文件路径 (请确保文件在当前目录或修改路径)
INPUT_FILE = "weekly_with_positions_share_vs_rank.csv"


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
    required_columns = ["season", "week", "pos_share", "pos_rank", "fan_rank"]
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


def calculate_weekly_correlation(
    df: pd.DataFrame,
    method_col: str,
    truth_col: str = "fan_rank",
    min_contestants: int = 4,
) -> pd.DataFrame:
    """
    计算每一周的 Spearman 相关系数

    Args:
        df: 数据框
        method_col: 待评估的方法排名列 (例如 'pos_share' 或 'pos_rank')
        truth_col: 真实排名列 (默认为 'fan_rank')
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


def compare_methods(df_corr_a: pd.DataFrame, df_corr_b: pd.DataFrame) -> Dict:
    """
    对比两种方法的相关系数结果
    """
    # 合并两组结果，确保只比较同一周的数据 (对齐)
    merged = pd.merge(
        df_corr_a, df_corr_b, on=["season", "week"], suffixes=("_share", "_rank")
    )

    # 计算差异 (Share模型 - 官方Rank模型)
    # rho 越高越好 (越接近 1 说明越准确)
    merged["diff"] = merged["rho_share"] - merged["rho_rank"]

    stats = {
        "n_weeks_compared": len(merged),
        "mean_rho_share": merged["rho_share"].mean(),
        "mean_rho_rank": merged["rho_rank"].mean(),
        "mean_diff": merged["diff"].mean(),
        "median_diff": merged["diff"].median(),
        "win_rate_share": (merged["diff"] > 0).mean(),  # Share 模型胜出的比例
        "win_rate_rank": (merged["diff"] < 0).mean(),  # Rank 模型胜出的比例
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
    print(f"2. Direct Head-to-Head Comparison:")
    print(f"   - Mean Improvement: {stats['mean_diff']:.4f}")
    print(f"   - Median Improvement: {stats['median_diff']:.4f}")
    print("-" * 30)
    print(f"3. Win Rate (Which model was more accurate per week?):")
    print(f"   - Share Model Wins: {stats['win_rate_share']*100:.1f}%")
    print(f"   - Rank Model Wins:  {stats['win_rate_rank']*100:.1f}%")
    print("=" * 50)

    if stats["mean_diff"] > 0:
        print("✅ CONCLUSION: The Share-based model is more accurate.")
        print("   It better reflects the true public sentiment (Fan Rank).")
    else:
        print("❌ CONCLUSION: The Official Rank-based model is more accurate.")


def main():
    # 1. 加载数据
    try:
        df = load_and_validate_data(INPUT_FILE)
    except Exception as e:
        print(e)
        return

    # 2. 计算每种方法的每周表现
    print("Calculating correlations for Share Model...")
    corr_share = calculate_weekly_correlation(
        df, method_col="pos_share", truth_col="fan_rank"
    )

    print("Calculating correlations for Rank Model...")
    corr_rank = calculate_weekly_correlation(
        df, method_col="pos_rank", truth_col="fan_rank"
    )

    # 3. 对比两种方法
    print("Comparing methods...")
    stats, comparison_df = compare_methods(corr_share, corr_rank)

    # 4. 输出报告
    print_report(stats)

    # (可选) 保存详细对比结果，方便画图
    # comparison_df.to_csv("model_comparison_results.csv", index=False)
    # print("\nDetailed results saved to 'model_comparison_results.csv'")


if __name__ == "__main__":
    main()
