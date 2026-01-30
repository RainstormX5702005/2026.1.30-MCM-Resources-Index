import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from configs.config import DATA_DIR, OUTPUT_DIR


def plot_scatter(df: pd.DataFrame):

    # 提取所有评委评分列
    week_judge_cols = [
        col for col in df.columns if "week" in col and "judge" in col and "score" in col
    ]

    # 重塑数据：将week_judge_score分离
    data_for_plot = []
    for idx, row in df.iterrows():
        for col in week_judge_cols:
            score = row[col]
            if pd.notna(score) and score > 0:  # 只保留有效分数
                # 解析列名，例如 week1_judge1_score -> week=1, judge=1
                parts = col.split("_")
                week = int(parts[0].replace("week", ""))
                judge = int(parts[1].replace("judge", ""))

                data_for_plot.append(
                    {
                        "celebrity_name": row["celebrity_name"],
                        "season": row["season"],
                        "week": week,
                        "judge": judge,
                        "score": score,
                    }
                )

    plot_df = pd.DataFrame(data_for_plot)

    # 获取所有独特的seasons
    seasons = sorted(plot_df["season"].unique())

    # 为每个season创建一个散点图
    for season in seasons:
        season_data = plot_df[plot_df["season"] == season]

        figure, ax = plt.subplots(figsize=(14, 8))

        # 按评委分组画散点图和折线图，每个评委用不同颜色
        judges = sorted(season_data["judge"].unique())
        colors = sns.color_palette("husl", len(judges))

        for judge, color in zip(judges, colors):
            judge_data = season_data[season_data["judge"] == judge]

            # 计算每周的平均分，用于连线
            week_avg = judge_data.groupby("week")["score"].mean().reset_index()
            week_avg = week_avg.sort_values("week")

            # 画所有散点（半透明）
            ax.scatter(
                judge_data["week"],
                judge_data["score"],
                color=color,
                alpha=0.3,
                s=50,
                edgecolors="none",
            )

            # 画平均分的折线和散点（突出显示）
            ax.plot(
                week_avg["week"],
                week_avg["score"],
                color=color,
                linewidth=2.5,
                alpha=0.8,
                marker="o",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label=f"Judge {judge}",
            )

        ax.set_xlabel("Week", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Judge Scores by Week - Season {season}",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )
        ax.legend(
            title="Judge",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            shadow=True,
        )
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 11)  # 分数范围通常是0-10

        # 设置x轴为整数
        weeks = sorted(season_data["week"].unique())
        ax.set_xticks(weeks)

        plt.tight_layout()

        # 保存图片
        output_path = OUTPUT_DIR / f"season_{season}_judge_scores_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

        plt.close()

    print(f"Total {len(seasons)} scatter plots created for seasons: {seasons}")


def plot_standardized_normal_distribution(df: pd.DataFrame, save_path: str):
    """绘制所有评分的标准正态分布变换直方图"""

    # 提取所有评委评分列
    week_judge_cols = [
        col for col in df.columns if "week" in col and "judge" in col and "score" in col
    ]

    # 收集所有有效评分
    all_scores = []
    for col in week_judge_cols:
        scores = df[col].dropna()
        scores = scores[scores > 0]  # 只保留有效分数
        all_scores.extend(scores.tolist())

    all_scores = np.array(all_scores)

    # 计算统计信息
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores, ddof=1)  # 样本标准差

    # 标准正态分布变换: z = (x - mean) / std
    standardized_scores = (all_scores - mean_score) / std_score

    # 创建包含两个子图的图形
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ===== 左侧子图：直方图 =====
    # 绘制直方图，使用density=False以获取频数
    n, bins, patches = ax1.hist(
        standardized_scores,
        bins=50,
        density=False,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
        label="Frequency (Count)",
    )

    # 创建右侧y轴用于显示密度
    ax1_right = ax1.twinx()

    # 在右侧y轴上绘制密度曲线（标准正态分布）
    x = np.linspace(standardized_scores.min(), standardized_scores.max(), 1000)
    # 将密度转换为频数以匹配左侧y轴的比例
    bin_width = bins[1] - bins[0]
    density_to_count = len(all_scores) * bin_width
    ax1_right.plot(
        x,
        stats.norm.pdf(x, 0, 1) * density_to_count,
        "r-",
        linewidth=2.5,
        label="Standard Normal Distribution (Scaled)",
    )

    # 添加统计信息文本
    stats_text = (
        f"Original Mean: {mean_score:.2f}\n"
        f"Original Std: {std_score:.2f}\n"
        f"Total Scores: {len(all_scores)}\n"
        f"Transformed Mean: {np.mean(standardized_scores):.4f}\n"
        f"Transformed Std: {np.std(standardized_scores, ddof=1):.4f}"
    )
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax1.set_xlabel("Standardized Score (Z-score)", fontsize=13, fontweight="bold")
    ax1.set_ylabel(
        "Count (Frequency)", fontsize=13, fontweight="bold", color="steelblue"
    )
    ax1_right.set_ylabel("Density", fontsize=13, fontweight="bold", color="red")

    # 设置y轴标签颜色
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1_right.tick_params(axis="y", labelcolor="red")

    ax1.set_title(
        "Histogram with Normal Distribution Curve",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=10,
        frameon=True,
        shadow=True,
    )

    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    # ===== 右侧子图：QQ图 =====
    # 添加小噪声以展示重叠的点
    standardized_scores_jittered = standardized_scores.copy()
    jitter_amount = np.random.normal(0, 0.02, len(standardized_scores))

    # 绘制QQ图
    stats.probplot(standardized_scores, dist="norm", plot=ax2)

    # 美化QQ图的点
    ax2.get_lines()[0].set_markerfacecolor("steelblue")
    ax2.get_lines()[0].set_markeredgecolor("black")
    ax2.get_lines()[0].set_markersize(4)
    ax2.get_lines()[0].set_alpha(0.5)
    ax2.get_lines()[0].set_markeredgewidth(0.3)

    # 参考线
    ax2.get_lines()[1].set_color("red")
    ax2.get_lines()[1].set_linewidth(2.5)
    ax2.get_lines()[1].set_zorder(5)

    ax2.set_xlabel("Theoretical Quantiles", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Sample Quantiles", fontsize=13, fontweight="bold")
    ax2.set_title(
        "Q-Q Plot (Normality Check)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    # 添加QQ图说明文本
    qq_text = (
        "If points lie on the red line,\n"
        "data follows normal distribution.\n"
        "Overplotted points indicate\n"
        "multiple samples at same position."
    )
    ax2.text(
        0.05,
        0.95,
        qq_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
    )

    # 整体标题
    figure.suptitle(
        "Standard Normal Distribution Analysis of All Judge Scores",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved standardized distribution plot: {save_path}")

    plt.close()


def main():
    file_path = OUTPUT_DIR / "processed_data.csv"
    df = pd.read_csv(file_path, sep=",", header=0, encoding="utf-8")
    save_path = str(OUTPUT_DIR / "figures" / "standard_gauss_dist.png")
    plot_standardized_normal_distribution(df, save_path=save_path)


if __name__ == "__main__":
    main()
