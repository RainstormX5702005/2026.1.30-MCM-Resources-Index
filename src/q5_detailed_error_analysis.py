"""
详细错误分析：将预测错误与原始特征数据结合
帮助理解为什么某些预测会失败
"""

import pandas as pd
import numpy as np
from pathlib import Path

from configs.config import OUTPUT_DIR


def load_all_data():
    """加载所有相关数据"""
    # 加载预测结果
    combined_df = pd.read_csv(OUTPUT_DIR / "question5_res" / "q5_combined.csv")

    # 加载原始特征数据
    featured_df = pd.read_csv(OUTPUT_DIR / "q4_featured_data.csv")

    # 加载错误分析
    errors_pct = pd.read_csv(OUTPUT_DIR / "question5_res" / "prediction_errors_pct.csv")

    errors_rank = pd.read_csv(
        OUTPUT_DIR / "question5_res" / "prediction_errors_rank.csv"
    )

    return combined_df, featured_df, errors_pct, errors_rank


def match_errors_with_features(errors_df, featured_df, prediction_type):
    """
    将错误预测与原始特征数据匹配
    """
    matched_data = []

    for idx, row in errors_df.iterrows():
        season = row["Season"]
        week = row["Week"]
        celebrity_name = row["Celebrity_Name"]

        matching_rows = featured_df[
            (featured_df["season"] == season)
            & (featured_df["celebrity_name"] == celebrity_name)
        ]

        if len(matching_rows) == 0:
            print(f"Warning: No match found for {celebrity_name} in season {season}")
            continue

        feature_row = matching_rows.iloc[0]

        merged_info = {
            "Season": season,
            "Week": week,
            "Celebrity_Name": celebrity_name,
            "Prediction_Type": prediction_type,
            "Predicted_Elim_Probability": row["Predicted_Elim_Probability"],
            "Actual_Eliminated": row["Actual_Eliminated"],
            "Prediction_Result": row["Prediction_Result"],
            # 添加特征信息
            "celebrity_age": feature_row.get("celebrity_age_during_season", np.nan),
            "celebrity_industry": feature_row.get("celebrity_industry", "Unknown"),
            "gender": feature_row.get("gender", np.nan),
            "is_final_reached": feature_row.get("is_final_reached", np.nan),
            "is_final_awarded": feature_row.get("is_final_awarded", np.nan),
            "relative_placement": feature_row.get("relative_placement", np.nan),
            "participated_ratio": feature_row.get("participated_ratio", np.nan),
        }

        # 添加该周的表现数据
        week_col_prefix = f"week{week}_"
        for col in featured_df.columns:
            if col.startswith(week_col_prefix):
                merged_info[col] = feature_row.get(col, np.nan)

        matched_data.append(merged_info)

    return pd.DataFrame(matched_data)


def analyze_error_patterns(errors_with_features_df):
    """分析错误预测的模式"""
    print("\n" + "=" * 80)
    print("错误预测模式分析")
    print("=" * 80)

    # 按预测结果类型分组
    fp_errors = errors_with_features_df[
        errors_with_features_df["Prediction_Result"] == "FP"
    ]
    fn_errors = errors_with_features_df[
        errors_with_features_df["Prediction_Result"] == "FN"
    ]

    print(f"\nFalse Positive (预测淘汰但实际未淘汰): {len(fp_errors)}")
    if len(fp_errors) > 0:
        print("\n--- False Positive 特征统计 ---")
        print(f"平均预测淘汰概率: {fp_errors['Predicted_Elim_Probability'].mean():.4f}")
        print(f"平均相对排名: {fp_errors['relative_placement'].mean():.4f}")
        print(f"平均参与比例: {fp_errors['participated_ratio'].mean():.4f}")
        print(f"到达决赛比例: {fp_errors['is_final_reached'].mean():.4f}")
        print("\n行业分布:")
        print(fp_errors["celebrity_industry"].value_counts())

    print(f"\n\nFalse Negative (预测未淘汰但实际淘汰): {len(fn_errors)}")
    if len(fn_errors) > 0:
        print("\n--- False Negative 特征统计 ---")
        print(f"平均预测淘汰概率: {fn_errors['Predicted_Elim_Probability'].mean():.4f}")
        print(f"平均相对排名: {fn_errors['relative_placement'].mean():.4f}")
        print(f"平均参与比例: {fn_errors['participated_ratio'].mean():.4f}")
        print(f"到达决赛比例: {fn_errors['is_final_reached'].mean():.4f}")
        print("\n行业分布:")
        print(fn_errors["celebrity_industry"].value_counts())


def create_detailed_error_report(errors_with_features_df, output_path):
    """创建详细的错误报告"""
    # 选择重要的列
    important_cols = [
        "Season",
        "Week",
        "Celebrity_Name",
        "Prediction_Type",
        "Predicted_Elim_Probability",
        "Actual_Eliminated",
        "Prediction_Result",
        "celebrity_age",
        "celebrity_industry",
        "gender",
        "is_final_reached",
        "is_final_awarded",
        "relative_placement",
        "participated_ratio",
    ]

    # 添加周数据列
    week_cols = [
        col for col in errors_with_features_df.columns if col.startswith("week")
    ]
    important_cols.extend(week_cols)

    # 筛选存在的列
    available_cols = [
        col for col in important_cols if col in errors_with_features_df.columns
    ]

    report_df = errors_with_features_df[available_cols].copy()

    # 排序：按预测概率降序
    report_df = report_df.sort_values("Predicted_Elim_Probability", ascending=False)

    # 保存
    report_df.to_csv(output_path, index=False)
    print(f"\n详细错误报告已保存至: {output_path}")


def main():
    print("=" * 80)
    print("详细错误分析：结合特征数据")
    print("=" * 80)

    # 加载数据
    print("\n加载数据...")
    combined_df, featured_df, errors_pct, errors_rank = load_all_data()

    output_dir = OUTPUT_DIR / "question5_res"

    # ========== 处理百分比方法的错误 ==========
    print("\n" + "=" * 80)
    print("百分比方法 (Season 3-27) 错误分析")
    print("=" * 80)

    if len(errors_pct) > 0:
        pct_errors_with_features = match_errors_with_features(
            errors_pct, featured_df, "pct"
        )

        analyze_error_patterns(pct_errors_with_features)

        create_detailed_error_report(
            pct_errors_with_features, output_dir / "detailed_error_report_pct.csv"
        )

        # 打印一些具体案例
        print("\n" + "-" * 80)
        print("具体案例 (前5个):")
        print("-" * 80)
        for idx, row in pct_errors_with_features.head(5).iterrows():
            print(f"\n案例 {idx+1}:")
            print(
                f"  赛季 {row['Season']} 第 {row['Week']} 周 - {row['Celebrity_Name']}"
            )
            print(f"  预测淘汰概率: {row['Predicted_Elim_Probability']:.4f}")
            print(f"  实际是否淘汰: {row['Actual_Eliminated']}")
            print(f"  错误类型: {row['Prediction_Result']}")
            print(f"  行业: {row.get('celebrity_industry', 'Unknown')}")
            print(f"  最终排名: {row.get('relative_placement', 'N/A')}")
    else:
        print("没有错误记录！")

    # ========== 处理排名方法的错误 ==========
    print("\n" + "=" * 80)
    print("排名方法 (其它赛季) 错误分析")
    print("=" * 80)

    if len(errors_rank) > 0:
        rank_errors_with_features = match_errors_with_features(
            errors_rank, featured_df, "rank"
        )

        analyze_error_patterns(rank_errors_with_features)

        create_detailed_error_report(
            rank_errors_with_features, output_dir / "detailed_error_report_rank.csv"
        )

        # 打印一些具体案例
        print("\n" + "-" * 80)
        print("具体案例 (前5个):")
        print("-" * 80)
        for idx, row in rank_errors_with_features.head(5).iterrows():
            print(f"\n案例 {idx+1}:")
            print(
                f"  赛季 {row['Season']} 第 {row['Week']} 周 - {row['Celebrity_Name']}"
            )
            print(f"  预测淘汰概率: {row['Predicted_Elim_Probability']:.4f}")
            print(f"  实际是否淘汰: {row['Actual_Eliminated']}")
            print(f"  错误类型: {row['Prediction_Result']}")
            print(f"  行业: {row.get('celebrity_industry', 'Unknown')}")
            print(f"  最终排名: {row.get('relative_placement', 'N/A')}")
    else:
        print("没有错误记录！")

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
