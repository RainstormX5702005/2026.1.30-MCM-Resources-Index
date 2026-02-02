"""
准确性检验：基于后验概率p值进行预测准确性分析
- 绘制p值图，纵轴是预测的淘汰概率，横轴是参赛者编号
- 区分实际被淘汰和未被淘汰的点
- 标记预测错误的点（高p值但未淘汰，或低p值但被淘汰）
- 分别为基于百分比的方法（Season 3-27）和基于排名的方法（其它赛季）绘制图像
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from configs.config import OUTPUT_DIR

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_data():
    """加载预测数据"""
    combined_df = pd.read_csv(
        OUTPUT_DIR / "question5_res" / "q5_combined.csv",
        dtype={
            "Season": "int64",
            "Week": "int64",
            "Celebrity_Name": "str",
            "Predicted_Elim_Probability": "float64",
            "Actual_Eliminated": "bool",
            "Prediction_Type": "str",
        },
    )
    return combined_df


def classify_predictions(df, threshold=0.5):
    """
    分类预测结果
    threshold: 判定为淘汰的概率阈值

    返回分类结果：
    - True Positive (TP): 预测淘汰且实际淘汰
    - False Positive (FP): 预测淘汰但实际未淘汰 (预测错误)
    - True Negative (TN): 预测未淘汰且实际未淘汰
    - False Negative (FN): 预测未淘汰但实际淘汰 (预测错误)
    """
    df = df.copy()
    df["Predicted_Eliminated"] = df["Predicted_Elim_Probability"] >= threshold

    df["Prediction_Result"] = "Unknown"

    # True Positive: 预测淘汰 + 实际淘汰
    mask_tp = (df["Predicted_Eliminated"] == True) & (df["Actual_Eliminated"] == True)
    df.loc[mask_tp, "Prediction_Result"] = "TP"

    # False Positive: 预测淘汰 + 实际未淘汰 (错误)
    mask_fp = (df["Predicted_Eliminated"] == True) & (df["Actual_Eliminated"] == False)
    df.loc[mask_fp, "Prediction_Result"] = "FP"

    # True Negative: 预测未淘汰 + 实际未淘汰
    mask_tn = (df["Predicted_Eliminated"] == False) & (df["Actual_Eliminated"] == False)
    df.loc[mask_tn, "Prediction_Result"] = "TN"

    # False Negative: 预测未淘汰 + 实际淘汰 (错误)
    mask_fn = (df["Predicted_Eliminated"] == False) & (df["Actual_Eliminated"] == True)
    df.loc[mask_fn, "Prediction_Result"] = "FN"

    return df


def calculate_accuracy_metrics(df):
    """计算准确率指标"""
    result_counts = df["Prediction_Result"].value_counts()

    tp = result_counts.get("TP", 0)
    fp = result_counts.get("FP", 0)
    tn = result_counts.get("TN", 0)
    fn = result_counts.get("FN", 0)

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    # 精确率：预测为淘汰的样本中，实际淘汰的比例
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # 召回率：实际淘汰的样本中，被正确预测的比例
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1分数
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics = {
        "Total": total,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    return metrics


def plot_p_value_analysis(df, prediction_type, output_dir, threshold=0.5):
    """
    绘制p值分析图

    参数：
    - df: 数据框
    - prediction_type: 'pct' 或 'rank'
    - output_dir: 输出目录
    - threshold: 淘汰概率阈值
    """
    # 按预测概率排序，方便可视化
    df_sorted = df.sort_values(
        "Predicted_Elim_Probability", ascending=False
    ).reset_index(drop=True)
    df_sorted["Index"] = range(len(df_sorted))

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 8))

    # 分类数据
    # 实际被淘汰的
    actual_elim = df_sorted[df_sorted["Actual_Eliminated"] == True]
    actual_not_elim = df_sorted[df_sorted["Actual_Eliminated"] == False]

    # 预测错误的点
    errors = df_sorted[df_sorted["Prediction_Result"].isin(["FP", "FN"])]
    fp_errors = df_sorted[df_sorted["Prediction_Result"] == "FP"]  # 高p但未淘汰
    fn_errors = df_sorted[df_sorted["Prediction_Result"] == "FN"]  # 低p但淘汰

    # 绘制所有点
    # 实际被淘汰的点（红色）
    ax.scatter(
        actual_elim["Index"],
        actual_elim["Predicted_Elim_Probability"],
        c="red",
        alpha=0.5,
        s=30,
        label="Actually Eliminated",
        marker="o",
    )

    # 实际未被淘汰的点（蓝色）
    ax.scatter(
        actual_not_elim["Index"],
        actual_not_elim["Predicted_Elim_Probability"],
        c="blue",
        alpha=0.5,
        s=30,
        label="Not Eliminated",
        marker="o",
    )

    # 特别标注预测错误的点
    # False Positive: 预测淘汰但实际未淘汰（蓝色叉）
    if len(fp_errors) > 0:
        ax.scatter(
            fp_errors["Index"],
            fp_errors["Predicted_Elim_Probability"],
            c="blue",
            marker="x",
            s=100,
            linewidths=2,
            label=f"FP: Pred Elim but Not (n={len(fp_errors)})",
            zorder=5,
        )

    # False Negative: 预测未淘汰但实际淘汰（红色叉）
    if len(fn_errors) > 0:
        ax.scatter(
            fn_errors["Index"],
            fn_errors["Predicted_Elim_Probability"],
            c="red",
            marker="x",
            s=100,
            linewidths=2,
            label=f"FN: Pred Not Elim but Was (n={len(fn_errors)})",
            zorder=5,
        )

    # 添加阈值线
    ax.axhline(
        y=threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold}",
        alpha=0.7,
    )

    # 计算并显示准确率指标
    metrics = calculate_accuracy_metrics(df)

    # 设置标题和标签
    method_name = (
        "Percentage-based (Season 3-27)"
        if prediction_type == "pct"
        else "Rank-based (Other Seasons)"
    )
    ax.set_title(
        f"Elimination Probability (p-value) Analysis - {method_name}\n"
        + f'Accuracy: {metrics["Accuracy"]:.3f} | Precision: {metrics["Precision"]:.3f} | '
        + f'Recall: {metrics["Recall"]:.3f} | F1: {metrics["F1"]:.3f}\n'
        + f'Total: {metrics["Total"]} | Errors: {metrics["FP"] + metrics["FN"]} '
        + f'(FP: {metrics["FP"]}, FN: {metrics["FN"]})',
        fontsize=14,
        pad=20,
    )

    ax.set_xlabel(
        "Contestant Index (Sorted by Predicted Elimination Probability)", fontsize=12
    )
    ax.set_ylabel("Predicted Elimination Probability (p-value)", fontsize=12)

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    # 添加说明文本
    info_text = (
        f"High p-value → High elimination probability\n"
        f"Expected: Red points (actually eliminated) have HIGH p-values\n"
        f"Expected: Blue points (not eliminated) have LOW p-values\n"
        f"X marks indicate prediction errors"
    )
    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    # 保存图像
    filename = f"accuracy_check_{prediction_type}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")

    plt.close()

    return metrics


def plot_high_probability_focus(df, prediction_type, output_dir, high_p_threshold=0.7):
    """
    聚焦高淘汰概率区域的图
    专门展示那些被判别为会被淘汰且概率很高的点
    """
    # 筛选高概率的点
    high_p_df = df[df["Predicted_Elim_Probability"] >= high_p_threshold].copy()

    if len(high_p_df) == 0:
        print(f"No data with p >= {high_p_threshold} for {prediction_type}")
        return

    high_p_df = high_p_df.sort_values(
        "Predicted_Elim_Probability", ascending=False
    ).reset_index(drop=True)
    high_p_df["Index"] = range(len(high_p_df))

    fig, ax = plt.subplots(figsize=(14, 7))

    # 实际被淘汰的高概率点
    actual_elim = high_p_df[high_p_df["Actual_Eliminated"] == True]
    actual_not_elim = high_p_df[high_p_df["Actual_Eliminated"] == False]

    # 错误的点
    fp_errors = high_p_df[high_p_df["Prediction_Result"] == "FP"]

    # 绘制点
    ax.scatter(
        actual_elim["Index"],
        actual_elim["Predicted_Elim_Probability"],
        c="red",
        alpha=0.6,
        s=50,
        label="Actually Eliminated",
        marker="o",
    )

    ax.scatter(
        actual_not_elim["Index"],
        actual_not_elim["Predicted_Elim_Probability"],
        c="blue",
        alpha=0.6,
        s=50,
        label="Not Eliminated (Error)",
        marker="o",
    )

    # 标注错误点
    if len(fp_errors) > 0:
        ax.scatter(
            fp_errors["Index"],
            fp_errors["Predicted_Elim_Probability"],
            c="orange",
            marker="x",
            s=150,
            linewidths=3,
            label=f"False Positive (n={len(fp_errors)})",
            zorder=5,
        )

    # 计算高概率区域的准确率
    high_p_correct = len(actual_elim)
    high_p_total = len(high_p_df)
    high_p_accuracy = high_p_correct / high_p_total if high_p_total > 0 else 0

    method_name = (
        "Percentage-based (Season 3-27)"
        if prediction_type == "pct"
        else "Rank-based (Other Seasons)"
    )
    ax.set_title(
        f"High Elimination Probability Analysis (p ≥ {high_p_threshold}) - {method_name}\n"
        + f"High-p Accuracy: {high_p_accuracy:.3f} ({high_p_correct}/{high_p_total})\n"
        + f"These are contestants predicted to be eliminated with high confidence",
        fontsize=13,
        pad=15,
    )

    ax.set_xlabel("High-Probability Contestant Index", fontsize=11)
    ax.set_ylabel("Predicted Elimination Probability (p-value)", fontsize=11)

    ax.set_ylim(high_p_threshold - 0.05, 1.05)
    ax.axhline(
        y=high_p_threshold,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label=f"Threshold = {high_p_threshold}",
    )

    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()

    filename = f"accuracy_check_{prediction_type}_high_p.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")

    plt.close()


def export_error_analysis(df, prediction_type, output_dir):
    """导出预测错误的详细信息"""
    errors = df[df["Prediction_Result"].isin(["FP", "FN"])].copy()

    if len(errors) == 0:
        print(f"No errors found for {prediction_type}")
        return

    errors = errors.sort_values("Predicted_Elim_Probability", ascending=False)

    filename = f"prediction_errors_{prediction_type}.csv"
    errors.to_csv(output_dir / filename, index=False)
    print(f"Exported error analysis: {filename}")
    print(f"  Total errors: {len(errors)}")
    print(
        f"  False Positives (high p but not eliminated): {len(errors[errors['Prediction_Result']=='FP'])}"
    )
    print(
        f"  False Negatives (low p but eliminated): {len(errors[errors['Prediction_Result']=='FN'])}"
    )


def main():
    """主函数"""
    print("=" * 80)
    print("准确性检验：基于后验概率的预测准确性分析")
    print("=" * 80)

    # 加载数据
    print("\n加载数据...")
    combined_df = load_data()
    print(f"总数据量: {len(combined_df)}")

    # 输出目录
    output_dir = OUTPUT_DIR / "question5_res"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 设置阈值
    threshold = 0.5
    high_p_threshold = 0.7

    print(f"\n使用阈值: {threshold} (预测淘汰)")
    print(f"高概率阈值: {high_p_threshold}")

    # ========== 处理基于百分比的方法 (Season 3-27) ==========
    print("\n" + "=" * 80)
    print("基于百分比的方法 (Season 3-27)")
    print("=" * 80)

    pct_df = combined_df[
        (combined_df["Season"] >= 3)
        & (combined_df["Season"] <= 27)
        & (combined_df["Prediction_Type"] == "pct")
    ].copy()

    print(f"数据量: {len(pct_df)}")

    # 分类预测结果
    pct_df = classify_predictions(pct_df, threshold=threshold)

    # 计算准确率
    pct_metrics = calculate_accuracy_metrics(pct_df)
    print("\n准确率指标:")
    print(f"  总样本数: {pct_metrics['Total']}")
    print(f"  准确率 (Accuracy): {pct_metrics['Accuracy']:.4f}")
    print(f"  精确率 (Precision): {pct_metrics['Precision']:.4f}")
    print(f"  召回率 (Recall): {pct_metrics['Recall']:.4f}")
    print(f"  F1分数: {pct_metrics['F1']:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  TP (预测淘汰，实际淘汰): {pct_metrics['TP']}")
    print(f"  FP (预测淘汰，实际未淘汰): {pct_metrics['FP']}")
    print(f"  TN (预测未淘汰，实际未淘汰): {pct_metrics['TN']}")
    print(f"  FN (预测未淘汰，实际淘汰): {pct_metrics['FN']}")

    # 绘制图像
    print("\n绘制图像...")
    plot_p_value_analysis(pct_df, "pct", output_dir, threshold=threshold)
    plot_high_probability_focus(
        pct_df, "pct", output_dir, high_p_threshold=high_p_threshold
    )

    # 导出错误分析
    export_error_analysis(pct_df, "pct", output_dir)

    # ========== 处理基于排名的方法 (其它赛季) ==========
    print("\n" + "=" * 80)
    print("基于排名的方法 (其它赛季)")
    print("=" * 80)

    rank_df = combined_df[combined_df["Prediction_Type"] == "rank"].copy()

    print(f"数据量: {len(rank_df)}")
    print(f"赛季: {sorted(rank_df['Season'].unique())}")

    # 分类预测结果
    rank_df = classify_predictions(rank_df, threshold=threshold)

    # 计算准确率
    rank_metrics = calculate_accuracy_metrics(rank_df)
    print("\n准确率指标:")
    print(f"  总样本数: {rank_metrics['Total']}")
    print(f"  准确率 (Accuracy): {rank_metrics['Accuracy']:.4f}")
    print(f"  精确率 (Precision): {rank_metrics['Precision']:.4f}")
    print(f"  召回率 (Recall): {rank_metrics['Recall']:.4f}")
    print(f"  F1分数: {rank_metrics['F1']:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  TP (预测淘汰，实际淘汰): {rank_metrics['TP']}")
    print(f"  FP (预测淘汰，实际未淘汰): {rank_metrics['FP']}")
    print(f"  TN (预测未淘汰，实际未淘汰): {rank_metrics['TN']}")
    print(f"  FN (预测未淘汰，实际淘汰): {rank_metrics['FN']}")

    # 绘制图像
    print("\n绘制图像...")
    plot_p_value_analysis(rank_df, "rank", output_dir, threshold=threshold)
    plot_high_probability_focus(
        rank_df, "rank", output_dir, high_p_threshold=high_p_threshold
    )

    # 导出错误分析
    export_error_analysis(rank_df, "rank", output_dir)

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"\n基于百分比方法 (Season 3-27):")
    print(f"  准确率: {pct_metrics['Accuracy']:.4f}")
    print(f"  总错误数: {pct_metrics['FP'] + pct_metrics['FN']}")

    print(f"\n基于排名方法 (其它赛季):")
    print(f"  准确率: {rank_metrics['Accuracy']:.4f}")
    print(f"  总错误数: {rank_metrics['FP'] + rank_metrics['FN']}")

    print(f"\n所有图像已保存至: {output_dir}")
    print("\n完成！")


if __name__ == "__main__":
    main()
