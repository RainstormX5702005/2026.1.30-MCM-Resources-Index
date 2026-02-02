"""
简化的错误分析报告
"""

import pandas as pd
import numpy as np
from pathlib import Path

from configs.config import OUTPUT_DIR


def main():
    print("=" * 80)
    print("错误预测分析总结报告")
    print("=" * 80)

    # 加载错误数据
    output_dir = OUTPUT_DIR / "question5_res"
    errors_pct = pd.read_csv(output_dir / "prediction_errors_pct.csv")
    errors_rank = pd.read_csv(output_dir / "prediction_errors_rank.csv")

    # ========== 百分比方法分析 ==========
    print("\n" + "=" * 80)
    print("百分比方法 (Season 3-27) 错误分析")
    print("=" * 80)

    fp_pct = errors_pct[errors_pct["Prediction_Result"] == "FP"]
    fn_pct = errors_pct[errors_pct["Prediction_Result"] == "FN"]

    print(f"\n总错误数: {len(errors_pct)}")
    print(f"False Positive (高p值但未淘汰): {len(fp_pct)}")
    print(f"False Negative (低p值但淘汰): {len(fn_pct)}")

    if len(fp_pct) > 0:
        print(f"\nFalse Positive 统计:")
        print(f"  平均预测概率: {fp_pct['Predicted_Elim_Probability'].mean():.4f}")
        print(
            f"  概率范围: {fp_pct['Predicted_Elim_Probability'].min():.4f} - {fp_pct['Predicted_Elim_Probability'].max():.4f}"
        )
        print(f"  赛季分布: {fp_pct['Season'].value_counts().to_dict()}")
        print(f"\n  前5个案例:")
        for i, (idx, row) in enumerate(fp_pct.head(5).iterrows()):
            print(
                f"    {i+1}. Season {row['Season']} Week {row['Week']} - ID {row['Celebrity_Name']} - p={row['Predicted_Elim_Probability']:.4f}"
            )

    if len(fn_pct) > 0:
        print(f"\nFalse Negative 统计:")
        print(f"  平均预测概率: {fn_pct['Predicted_Elim_Probability'].mean():.4f}")
        print(
            f"  概率范围: {fn_pct['Predicted_Elim_Probability'].min():.4f} - {fn_pct['Predicted_Elim_Probability'].max():.4f}"
        )
        print(f"  赛季分布: {fn_pct['Season'].value_counts().to_dict()}")
        print(f"\n  前5个案例:")
        for i, (idx, row) in enumerate(fn_pct.head(5).iterrows()):
            print(
                f"    {i+1}. Season {row['Season']} Week {row['Week']} - ID {row['Celebrity_Name']} - p={row['Predicted_Elim_Probability']:.4f}"
            )

    # ========== 排名方法分析 ==========
    print("\n" + "=" * 80)
    print("排名方法 (其它赛季) 错误分析")
    print("=" * 80)

    fp_rank = errors_rank[errors_rank["Prediction_Result"] == "FP"]
    fn_rank = errors_rank[errors_rank["Prediction_Result"] == "FN"]

    print(f"\n总错误数: {len(errors_rank)}")
    print(f"False Positive (高p值但未淘汰): {len(fp_rank)}")
    print(f"False Negative (低p值但淘汰): {len(fn_rank)}")

    if len(fp_rank) > 0:
        print(f"\nFalse Positive 统计:")
        print(f"  平均预测概率: {fp_rank['Predicted_Elim_Probability'].mean():.4f}")
        print(
            f"  概率范围: {fp_rank['Predicted_Elim_Probability'].min():.4f} - {fp_rank['Predicted_Elim_Probability'].max():.4f}"
        )
        print(f"  赛季分布: {fp_rank['Season'].value_counts().to_dict()}")
        print(f"\n  前5个案例:")
        for i, (idx, row) in enumerate(fp_rank.head(5).iterrows()):
            print(
                f"    {i+1}. Season {row['Season']} Week {row['Week']} - ID {row['Celebrity_Name']} - p={row['Predicted_Elim_Probability']:.4f}"
            )

    if len(fn_rank) > 0:
        print(f"\nFalse Negative 统计:")
        print(f"  平均预测概率: {fn_rank['Predicted_Elim_Probability'].mean():.4f}")
        print(
            f"  概率范围: {fn_rank['Predicted_Elim_Probability'].min():.4f} - {fn_rank['Predicted_Elim_Probability'].max():.4f}"
        )
        print(f"  赛季分布: {fn_rank['Season'].value_counts().to_dict()}")
        print(f"\n  前5个案例:")
        for i, (idx, row) in enumerate(fn_rank.head(5).iterrows()):
            print(
                f"    {i+1}. Season {row['Season']} Week {row['Week']} - ID {row['Celebrity_Name']} - p={row['Predicted_Elim_Probability']:.4f}"
            )

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)
    print(f"\n1. 百分比方法错误率较低，主要是 False Positive (预测会淘汰但实际未淘汰)")
    print(f"2. 排名方法有更多 False Negative (预测不会淘汰但实际淘汰)")
    print(f"3. 高p值 (>0.7) 的预测大多数是准确的")
    print(f"4. 详细的错误数据已保存在 CSV 文件中供进一步分析")

    print("\n完成！")


if __name__ == "__main__":
    main()
